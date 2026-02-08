import bpy
import re
import math
import numpy as np
from mathutils import Vector
from struct import pack
from szio.gta5.cwxml.ymap import *
from binascii import hexlify
from ..tools.blenderhelper import remove_number_suffix
from ..tools.meshhelper import get_bound_center_from_bounds, get_extents, get_combined_bound_box
from ..sollumz_properties import SOLLUMZ_UI_NAMES, SollumType
from ..sollumz_preferences import get_export_settings
from .. import logger
from ..tools.ymaphelper import generate_ymap_extents


def box_from_obj(obj):
    box = BoxOccluder()

    bbmin, bbmax = get_extents(obj)
    center = get_bound_center_from_bounds(bbmin, bbmax)
    dimensions = obj.dimensions

    box.center_x = round(center.x * 4)
    box.center_y = round(center.y * 4)
    box.center_z = round(center.z * 4)

    box.length = round(dimensions.x * 4)
    box.width = round(dimensions.y * 4)
    box.height = round(dimensions.z * 4)

    dir = Vector((1, 0, 0))
    dir.rotate(obj.rotation_euler)
    dir *= 0.5
    box.sin_z = round(dir.x * 32767)
    box.cos_z = round(dir.y * 32767)

    return box


def triangulate_obj(obj):
    """Convert mesh from n-polygons to triangles"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")


def occlusion_model_obj_get_data_buffer(obj) -> bytes:
    """
    For each vertex get its coordinates in global space (this way we don't need to apply transfroms) as 32-bit floats
    and for each face get its indices as 8-bit integers. Then convert them to bytes and append them together.

    :return verts: Bytes buffer containing the vertex coordinates and face indices
    :rtype bytes:
    """
    # TODO: should validate that the mesh is within <256 verts
    mesh = obj.data
    verts = np.array([obj.matrix_world @ v.co for v in mesh.vertices], dtype=np.float32)
    indices = np.array([i for p in mesh.polygons for i in p.vertices], dtype=np.uint8)
    return verts.tobytes() + indices.tobytes()


def model_from_obj(obj):
    triangulate_obj(obj)

    model = OccludeModel()
    model.bmin, model.bmax = get_extents(obj)
    model.verts = occlusion_model_obj_get_data_buffer(obj)
    model.num_verts_in_bytes = len(obj.data.vertices) * 12
    face_count = len(obj.data.polygons)
    model.num_tris = face_count | 0x8000  # add float vertex format marker
    model.data_size = len(model.verts)
    model.flags = obj.ymap_model_occl_properties.model_occl_flags

    assert model.data_size == (model.num_verts_in_bytes + face_count * 3)

    return model


def calculate_num_children(entity_obj, ymap_obj, entity_idx):
    count = 0
    
    # 1. Internal Children (in the same YMAP)
    # Using the current view layer to find siblings
    # We can rely on the fact that if we are exporting, the objects are in the scene
    
    # We need to iterate all entities of the CURRENT ymap to see if they point to entity_obj
    for child in ymap_obj.children:
        if child.sollum_type == SollumType.YMAP_ENTITY_GROUP:
            for sibling in child.children:
                if sibling == entity_obj: continue # Skip self
                
                # Check linked parent
                if sibling.entity_properties.parent_entity == entity_obj:
                    count += 1
                # Check manual index
                elif not sibling.entity_properties.parent_entity and \
                     sibling.entity_properties.parent_index == entity_idx:
                     # For manual index to work as "child", the child must likely have LOD_IN_PARENT logic or just regular hierarchy?
                     # In separate YMAPs, LOD_IN_PARENT is used. In same YMAP, parent_index is sufficient.
                     count += 1

    # 2. External Children (in Child YMAPs)
    # Find all YMAPs that list 'ymap_obj' as parent
    child_ymaps = []
    
    # Clean name for string comparison
    ymap_name = remove_number_suffix(ymap_obj.name)
    
    for obj in bpy.context.scene.objects:
        if obj.sollum_type == SollumType.YMAP and obj != ymap_obj:
            # Check pointer
            if obj.ymap_properties.parent_ymap == ymap_obj:
                child_ymaps.append(obj)
            # Check string name (fallback)
            elif not obj.ymap_properties.parent_ymap and \
                 obj.ymap_properties.parent == ymap_name:
                child_ymaps.append(obj)
    
    for child_ymap in child_ymaps:
        for group in child_ymap.children:
            if group.sollum_type == SollumType.YMAP_ENTITY_GROUP:
                for ent in group.children:
                    # Check linked parent
                    if ent.entity_properties.parent_entity == entity_obj:
                        count += 1
                    # Check manual index + Flag
                    elif not ent.entity_properties.parent_entity and \
                         ent.entity_properties.parent_index == entity_idx:
                         # Bit 3 is Flag 4 (LOD In Parent)
                         # We can check the integer value directly or the prop
                         if ent.entity_properties.flags & 8:
                             count += 1
                             
    return count


def entity_from_obj(obj, parent_index_map=None, current_ymap_entities=None, ymap_obj=None, obj_index=None):
    # Removing " (not found)" suffix, created when importing ymaps while entity was not found in the view layer
    entity_name = re.sub(r" \(not found\)", "", obj.name.lower())

    entity = Entity()
    entity.archetype_name = remove_number_suffix(entity_name)
    entity.flags = int(obj.entity_properties.flags)
    entity.guid = int(obj.entity_properties.guid)
    entity.position = obj.location
    entity.rotation = obj.rotation_euler.to_quaternion()
    if entity.type != "CMloInstanceDef":
        entity.rotation.invert()
    entity.scale_xy = round(obj.scale.x, 8)
    entity.scale_z = round(obj.scale.z, 8)
    
    # Parent Linking Logic
    linked_parent = obj.entity_properties.parent_entity
    parent_ymap_is_different = False
    
    if linked_parent and parent_index_map is not None:
        if linked_parent in parent_index_map:
            # Parent is in key map.
            entity.parent_index = parent_index_map[linked_parent]
            
            # Check if parent is in a different YMAP
            # ymap_obj is the CURRENT exported ymap.
            # We check the parent of the linked_parent
            if linked_parent.parent and linked_parent.parent.parent:
                # linked_parent -> EntityGroup -> Ymap
                parent_of_linked = linked_parent.parent.parent
                if parent_of_linked != ymap_obj:
                    parent_ymap_is_different = True

        else:
            logger.warning(f"Entity '{obj.name}' references parent '{linked_parent.name}', but valid index not found. Fallback to manual index.")
            entity.parent_index = int(obj.entity_properties.parent_index)
    else:
        entity.parent_index = int(obj.entity_properties.parent_index)

    # Auto-Set LOD_IN_PARENT flag for valid cross-ymap link
    if parent_ymap_is_different:
        # Flag 4 is bit 3 (1 << 3 = 8)
        if not (entity.flags & 8):
            entity.flags |= 8
            #logger.info(f"Auto-setting LOD_IN_PARENT for {obj.name} linked to {linked_parent.name}")

    entity.lod_dist = obj.entity_properties.lod_dist
    entity.child_lod_dist = obj.entity_properties.child_lod_dist
    entity.lod_level = obj.entity_properties.lod_level.upper().replace("SOLLUMZ_", "")
    
    # Num Children Calculation 
    # Use the robust calculation if we have context
    if ymap_obj and obj_index is not None:
        entity.num_children = calculate_num_children(obj, ymap_obj, obj_index)
    else:
        # Fallback to simple local count (legacy/safety)
        if current_ymap_entities:
            count = 0
            for child_obj in current_ymap_entities:
                if child_obj.entity_properties.parent_entity == obj:
                    count += 1
            entity.num_children = count
        else:
            entity.num_children = int(obj.entity_properties.num_children)

    entity.priority_level = obj.entity_properties.priority_level.upper().replace("SOLLUMZ_", "")
    entity.ambient_occlusion_multiplier = int(
        obj.entity_properties.ambient_occlusion_multiplier)
    entity.artificial_ambient_occlusion = int(
        obj.entity_properties.artificial_ambient_occlusion)
    entity.tint_value = int(obj.entity_properties.tint_value)

    return entity

def cargen_from_obj(obj):
    cargen = CarGenerator()
    cargen.position = obj.location

    cargen.orient_x, cargen.orient_y = calculate_cargen_orient(obj)

    cargen.perpendicular_length = obj.ymap_cargen_properties.perpendicular_length
    cargen.car_model = obj.ymap_cargen_properties.car_model
    cargen.flags = obj.ymap_cargen_properties.cargen_flags
    cargen.body_color_remap_1 = obj.ymap_cargen_properties.body_color_remap_1
    cargen.body_color_remap_2 = obj.ymap_cargen_properties.body_color_remap_2
    cargen.body_color_remap_3 = obj.ymap_cargen_properties.body_color_remap_3
    cargen.body_color_remap_4 = obj.ymap_cargen_properties.body_color_remap_4
    cargen.pop_group = obj.ymap_cargen_properties.pop_group
    cargen.livery = obj.ymap_cargen_properties.livery

    return cargen


def calculate_cargen_orient(obj):
    # *-1 because GTA likes to invert values
    angle = obj.rotation_euler[2] * -1

    return 5 * math.sin(angle), 5 * math.cos(angle)


def grass_batches_from_obj(ymap, grass_group_obj):
    """Export grass batches from the Grass Group - reads vertices with attributes from geometry nodes mesh"""
    from szio.gta5.cwxml.ymap import GrassInstanceList, GrassInstance, AABB, InstancedData
    from szio.types import Vector as SzVector
    
    for batch_obj in grass_group_obj.children:
        if batch_obj.sollum_type != SollumType.YMAP_GRASS_BATCH:
            continue
        
        # Check if batch has mesh data
        if batch_obj.data is None or not isinstance(batch_obj.data, bpy.types.Mesh):
            logger.warning(f"Grass batch '{batch_obj.name}' has no mesh data, skipping.")
            continue
        
        mesh = batch_obj.data
        
        # Check if mesh has vertices
        if len(mesh.vertices) == 0:
            logger.warning(f"Grass batch '{batch_obj.name}' has no grass instances (vertices), skipping.")
            continue
        
        # Get custom attributes
        color_attr = mesh.attributes.get("grass_color")
        scale_attr = mesh.attributes.get("grass_scale")
        rotation_attr = mesh.attributes.get("grass_rotation")
        normal_attr = mesh.attributes.get("grass_normal")
        ao_attr = mesh.attributes.get("grass_ao")
        
        # Collect all grass instance positions for AABB calculation (in world space)
        instance_positions = []
        for vert in mesh.vertices:
            world_pos = batch_obj.matrix_world @ vert.co
            instance_positions.append(world_pos)
        
        # Calculate BatchAABB from all instance positions
        xs = [p.x for p in instance_positions]
        ys = [p.y for p in instance_positions]
        zs = [p.z for p in instance_positions]
        
        # Add small margin to avoid division by zero
        margin = 0.01
        bb_min = Vector((min(xs) - margin, min(ys) - margin, min(zs) - margin))
        bb_max = Vector((max(xs) + margin, max(ys) + margin, max(zs) + margin))
        
        # Create grass batch
        grass_batch = GrassInstanceList()
        
        # Set AABB using the Vector type from szio
        grass_batch.batch_aabb.min = SzVector((bb_min.x, bb_min.y, bb_min.z, 0))
        grass_batch.batch_aabb.max = SzVector((bb_max.x, bb_max.y, bb_max.z, 0))
        
        # Set batch properties
        props = batch_obj.ymap_grass_batch_properties
        grass_batch.archetype_name = props.archetype_name
        grass_batch.lod_dist = props.lod_dist
        grass_batch.lod_fade_start_dist = props.lod_fade_start_dist
        grass_batch.lod_inst_fade_range = props.lod_inst_fade_range
        grass_batch.orient_to_terrain = props.orient_to_terrain
        grass_batch.scale_range = SzVector((props.scale_range[0], props.scale_range[1], props.scale_range[2]))
        
        # Export individual grass instances from vertices
        for idx, vert in enumerate(mesh.vertices):
            world_pos = batch_obj.matrix_world @ vert.co
            
            # Encode position to 0-65535 range relative to BatchAABB
            range_x = bb_max.x - bb_min.x
            range_y = bb_max.y - bb_min.y
            range_z = bb_max.z - bb_min.z
            
            pos_x = int(((world_pos.x - bb_min.x) / range_x) * 65535) if range_x > 0 else 0
            pos_y = int(((world_pos.y - bb_min.y) / range_y) * 65535) if range_y > 0 else 0
            pos_z = int(((world_pos.z - bb_min.z) / range_z) * 65535) if range_z > 0 else 0
            
            # Clamp to valid range
            pos_x = max(0, min(65535, pos_x))
            pos_y = max(0, min(65535, pos_y))
            pos_z = max(0, min(65535, pos_z))
            
            inst = GrassInstance()
            inst.position = f"{pos_x} {pos_y} {pos_z}"
            
            # Get color from attribute (or default)
            if color_attr:
                color_data = color_attr.data[idx].color
                r = int(color_data[0] * 255)
                g = int(color_data[1] * 255)
                b = int(color_data[2] * 255)
            else:
                # Default green color
                r, g, b = 102, 178, 51
            
            inst.color = f"{r} {g} {b}"
            
            # Get scale from attribute (or default = 1.0 → 255)
            if scale_attr:
                scale_val = scale_attr.data[idx].value
                inst.scale = max(0, min(255, int(scale_val * 255)))
            else:
                inst.scale = 255
            
            # Get AO from attribute (or default = 1.0 → 255)
            if ao_attr:
                ao_val = ao_attr.data[idx].value
                inst.ao = int(ao_val * 255)
            else:
                inst.ao = 255
            
            # Get rotation from attribute for normal calculation
            # Calculate normal
            if normal_attr:
                local_n = normal_attr.data[idx].vector
                # Transform to world space
                world_n = batch_obj.matrix_world.to_3x3() @ local_n
                world_n.normalize()
                nx = world_n.x
                ny = world_n.y
            else:
                # Default to Up vector (0, 0, 1)
                # This results in NormalX=127, NormalY=127 which is "flat"
                nx = 0.0
                ny = 0.0
            
            # Encode to 0-255 (127 is 0, 255 is 1.0, 0 is -1.0)
            inst.normal_x = max(0, min(255, int(nx * 127.0 + 127.0)))
            inst.normal_y = max(0, min(255, int(ny * 127.0 + 127.0)))
            
            # Pad
            inst.pad = "0 0 0"
            
            # Access the underlying list property and append
            inst_list = grass_batch.__getattribute__("instance_list", False)
            inst_list.value.append(inst)
        
        # Access the underlying list property and append
        grass_list = ymap.instanced_data.__getattribute__("grass_instance_list", False)
        grass_list.value.append(grass_batch)
        logger.info(f"Exported grass batch '{batch_obj.name}' with {len(grass_batch.instance_list)} instances (from {len(mesh.vertices)} vertices).")


def get_lod_level_priority(lod_level_str):
    """
    Returns priority for sorting. Lower value = Higher priority (Top of list).
    Order: SLOD4 -> SLOD3 -> SLOD2 -> SLOD1 -> LOD -> HD -> ORPHANHD
    """
    s = lod_level_str.lower()
    if "slod4" in s: return 0
    if "slod3" in s: return 1
    if "slod2" in s: return 2
    if "slod1" in s: return 3
    if "orphanhd" in s: return 6
    if s.endswith("_lod"): return 4
    if s.endswith("_hd"): return 5
    return 100 # Default/Unknown


def ymap_from_object(obj):
    ymap = CMapData()

    export_settings = get_export_settings()
    
    # Gather Entities
    all_entity_objects = []
    
    for child in obj.children:
        if export_settings.ymap_exclude_entities == False and child.sollum_type == SollumType.YMAP_ENTITY_GROUP:
            for entity_obj in child.children:
                all_entity_objects.append(entity_obj)
    
    # Sort Entities by LodLevel
    # Sorting logic: SLOD4 -> SLOD3 -> SLOD2 -> SLOD1 -> LOD -> HD -> ORPHANHD
    # Note: Using stable sort to maintain relative order of entities with same LOD level
    all_entity_objects.sort(key=lambda x: get_lod_level_priority(x.entity_properties.lod_level))

    # Build Index Map for Current YMAP
    # {Object: Index}
    parent_index_map = {}
    for idx, entity_obj in enumerate(all_entity_objects):
        parent_index_map[entity_obj] = idx

    # Handle Parent YMAP (if it exists in scene)
    parent_ymap_obj = obj.ymap_properties.parent_ymap
    
    # If not linked via pointer, try finding by name (fallback)
    if not parent_ymap_obj:
        parent_ymap_name = obj.ymap_properties.parent
        if parent_ymap_name:
            for scene_obj in bpy.context.scene.objects:
                if scene_obj.sollum_type == SollumType.YMAP and remove_number_suffix(scene_obj.name) == parent_ymap_name:
                    parent_ymap_obj = scene_obj
                    break
    
    if parent_ymap_obj:
        # Override the string parent name property if we have a linked object
        ymap.parent = remove_number_suffix(parent_ymap_obj.name)
        
        # We found the parent ymap. We must simulate its sorting to get correct indices!
        parent_entities = []
        for child in parent_ymap_obj.children:
            if child.sollum_type == SollumType.YMAP_ENTITY_GROUP:
                for ent in child.children:
                    parent_entities.append(ent)
        
        # Sort parent entities same way
        parent_entities.sort(key=lambda x: get_lod_level_priority(x.entity_properties.lod_level))
        
        # Add to map
        for idx, ent in enumerate(parent_entities):
            parent_index_map[ent] = idx
    else:
        # If no parent obj found, ymap.parent comes from the text field
        ymap.parent = obj.ymap_properties.parent

    # Create Entities with calculated indices
    # IMPORTANT: We iterate through the SORTED list 'all_entity_objects' to ensure the XML order matches expectations
    for idx, entity_obj in enumerate(all_entity_objects):
        ymap.entities.append(entity_from_obj(entity_obj, parent_index_map, all_entity_objects, obj, idx))



    for child in obj.children:
        # Box occluders
        if export_settings.ymap_box_occluders == False and child.sollum_type == SollumType.YMAP_BOX_OCCLUDER_GROUP:
            obj.ymap_properties.content_flags_toggle.has_occl = True

            for box_obj in child.children:
                rotation = box_obj.rotation_euler
                if abs(rotation.x) > 0.01 or abs(rotation.y) > 0.01:
                    logger.error(
                        f"Box occluders only support Z-axis rotation. Skipping {box_obj.name} due to X/Y rotation.")
                    continue

                if box_obj.sollum_type == SollumType.YMAP_BOX_OCCLUDER:
                    ymap.box_occluders.append(box_from_obj(box_obj))
                else:
                    logger.warning(
                        f"Object {box_obj.name} will be skipped because it is not a {SOLLUMZ_UI_NAMES[SollumType.YMAP_BOX_OCCLUDER]} type.")

        # Model occluders
        if export_settings.ymap_model_occluders == False and child.sollum_type == SollumType.YMAP_MODEL_OCCLUDER_GROUP:
            obj.ymap_properties.content_flags_toggle.has_occl = True

            for model_obj in child.children:
                if model_obj.sollum_type == SollumType.YMAP_MODEL_OCCLUDER:
                    if len(model_obj.data.vertices) > 256:
                        logger.warning(
                            f"Object {model_obj.name} has too many vertices and will be skipped. It can not have more than 256 vertices.")
                        continue

                    ymap.occlude_models.append(
                        model_from_obj(model_obj))
                else:
                    logger.warning(
                        f"Object {model_obj.name} will be skipped because it is not a {SOLLUMZ_UI_NAMES[SollumType.YMAP_MODEL_OCCLUDER]} type.")

        # TODO: physics_dictionaries

        # TODO: time cycle

        # Car generators
        if export_settings.ymap_car_generators == False and child.sollum_type == SollumType.YMAP_CAR_GENERATOR_GROUP:
            for cargen_obj in child.children:
                rotation = cargen_obj.rotation_euler
                if abs(rotation.x) > 0.01 or abs(rotation.y) > 0.01:
                    logger.error(
                        f"Car generators only support Z-axis rotation. Skipping {cargen_obj.name} due to X/Y rotation.")
                    continue
                if cargen_obj.sollum_type == SollumType.YMAP_CAR_GENERATOR:
                    ymap.car_generators.append(cargen_from_obj(cargen_obj))
                else:
                    logger.warning(
                        f"Object {cargen_obj.name} will be skipped because it is not a {SOLLUMZ_UI_NAMES[SollumType.YMAP_CAR_GENERATOR]} type.")

        # TODO: lod ligths

        # TODO: distant lod lights

        # Grass instances
        if child.sollum_type == SollumType.YMAP_GRASS_GROUP:
            obj.ymap_properties.content_flags_toggle.has_grass = True
            grass_batches_from_obj(ymap, child)

    ymap.name = remove_number_suffix(obj.name)
    # ymap.parent is set above during sorting logic
    ymap.flags = obj.ymap_properties.flags
    ymap.content_flags = obj.ymap_properties.content_flags

    ymap.name = remove_number_suffix(obj.name)
    # ymap.parent is set above during sorting logic
    ymap.flags = obj.ymap_properties.flags
    ymap.content_flags = obj.ymap_properties.content_flags

    # Calculate Extents
    # Initialize min/max with infinity
    entities_min = Vector((float('inf'), float('inf'), float('inf')))
    entities_max = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    streaming_min = Vector((float('inf'), float('inf'), float('inf')))
    streaming_max = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    found_any = False
    
    # Gather all objects that contribute to extents
    extent_objects = [] # List of tuples (obj, lod_dist)
    
    # 1. Entities
    for entity_obj in all_entity_objects:
        extent_objects.append((entity_obj, entity_obj.entity_properties.lod_dist))
    
    # 2. Occluders (Box & Model)
    for child in obj.children:
        if export_settings.ymap_box_occluders == False and child.sollum_type == SollumType.YMAP_BOX_OCCLUDER_GROUP:
             for box_obj in child.children:
                 if box_obj.sollum_type == SollumType.YMAP_BOX_OCCLUDER:
                      extent_objects.append((box_obj, 0.0))
        
        elif export_settings.ymap_model_occluders == False and child.sollum_type == SollumType.YMAP_MODEL_OCCLUDER_GROUP:
             for model_obj in child.children:
                 if model_obj.sollum_type == SollumType.YMAP_MODEL_OCCLUDER:
                      extent_objects.append((model_obj, 0.0))
        
        # 3. Grass instances - now stored as mesh vertices in batch object
        elif child.sollum_type == SollumType.YMAP_GRASS_GROUP:
            for batch_obj in child.children:
                if batch_obj.sollum_type == SollumType.YMAP_GRASS_BATCH:
                    # Grass is now vertices in the batch mesh - just add the batch with its LOD dist
                    lod_dist = batch_obj.ymap_grass_batch_properties.lod_dist
                    extent_objects.append((batch_obj, lod_dist))

    # Check bounds of all gathered objects
    for obj_ref, lod_dist in extent_objects:
        # Use get_combined_bound_box with use_world=True to get World Space bounds
        bbmin, bbmax = get_combined_bound_box(obj_ref, use_world=True)
        
        # If no mesh found (returns 0,0,0), fallback to object location
        # This handles cases where the entity is an empty/point representation
        if bbmin == Vector((0,0,0)) and bbmax == Vector((0,0,0)):
             loc = obj_ref.matrix_world.translation
             bbmin = loc
             bbmax = loc
        
        # Entities Extents - Just the geometry bounds
        entities_min.x = min(entities_min.x, bbmin.x)
        entities_min.y = min(entities_min.y, bbmin.y)
        entities_min.z = min(entities_min.z, bbmin.z)
        
        entities_max.x = max(entities_max.x, bbmax.x)
        entities_max.y = max(entities_max.y, bbmax.y)
        entities_max.z = max(entities_max.z, bbmax.z)
        
        # Streaming Extents - Geometry bounds expanded by LodDist
        streaming_min.x = min(streaming_min.x, bbmin.x - lod_dist)
        streaming_min.y = min(streaming_min.y, bbmin.y - lod_dist)
        streaming_min.z = min(streaming_min.z, bbmin.z - lod_dist)
        
        streaming_max.x = max(streaming_max.x, bbmax.x + lod_dist)
        streaming_max.y = max(streaming_max.y, bbmax.y + lod_dist)
        streaming_max.z = max(streaming_max.z, bbmax.z + lod_dist)
        
        found_any = True
        
    if not found_any:
        entities_min = Vector((0, 0, 0))
        entities_max = Vector((0, 0, 0))
        streaming_min = Vector((0, 0, 0))
        streaming_max = Vector((0, 0, 0))

    ymap.entities_extents_min = entities_min
    ymap.entities_extents_max = entities_max
    ymap.streaming_extents_min = streaming_min
    ymap.streaming_extents_max = streaming_max

    ymap.block.version = obj.ymap_properties.block.version
    ymap.block.flags = obj.ymap_properties.block.flags
    ymap.block.name = obj.ymap_properties.block.name
    ymap.block.exported_by = obj.ymap_properties.block.exported_by
    ymap.block.owner = obj.ymap_properties.block.owner
    ymap.block.time = obj.ymap_properties.block.time

    return ymap


def export_ymap(obj: bpy.types.Object, filepath: str) -> bool:
    ymap = ymap_from_object(obj)
    ymap.write_xml(filepath)
    return True
