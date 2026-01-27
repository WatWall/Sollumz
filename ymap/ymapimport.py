import math
import bpy
import numpy as np
from numpy.typing import NDArray
from mathutils import Vector, Euler
from ..sollumz_helper import duplicate_object_with_children, set_object_collection
from ..tools.ymaphelper import add_occluder_material, get_cargen_mesh
from ..sollumz_properties import SollumType
from ..sollumz_preferences import get_import_settings
from szio.gta5.cwxml import (
    CMapData,
    OccludeModel,
    YMAP,
)
from ..tools.blenderhelper import create_blender_object, create_empty_object
from ..tools.meshhelper import create_box
from .. import logger

# TODO: Make better?


def occlude_model_to_mesh_data(model: OccludeModel) -> tuple[NDArray, NDArray]:
    assert (model.num_tris & 0x8000) != 0, "Only float vertex format of occlude models is supported"

    num_verts_in_bytes = model.num_verts_in_bytes
    num_verts = num_verts_in_bytes // (4*3)  # sizeof(float)*3
    num_tris = model.num_tris & ~0x8000

    data = np.frombuffer(model.verts, dtype=np.uint8)
    verts = data[:num_verts_in_bytes].view(dtype=np.float32).reshape((num_verts, 3))
    faces = data[num_verts_in_bytes:].reshape((num_tris, 3))
    return verts, faces


def apply_entity_properties(obj, entity):
    obj.entity_properties.archetype_name = entity.archetype_name
    obj.entity_properties.flags = entity.flags
    obj.entity_properties.guid = entity.guid
    obj.entity_properties.parent_index = entity.parent_index
    obj.entity_properties.lod_dist = entity.lod_dist
    obj.entity_properties.child_lod_dist = entity.child_lod_dist
    obj.entity_properties.lod_level = "sollumz_" + entity.lod_level.lower()
    obj.entity_properties.num_children = entity.num_children
    obj.entity_properties.priority_level = "sollumz_" + entity.priority_level.lower()
    obj.entity_properties.ambient_occlusion_multiplier = entity.ambient_occlusion_multiplier
    obj.entity_properties.artificial_ambient_occlusion = entity.artificial_ambient_occlusion
    obj.entity_properties.tint_value = entity.tint_value
    if entity.type != "CMloInstanceDef":
        # Entities in YMAPs need rotation inverted
        entity.rotation.invert()
    obj.matrix_world = entity.rotation.to_matrix().to_4x4()
    obj.location = entity.position
    obj.scale = Vector((entity.scale_xy, entity.scale_xy, entity.scale_z))


def entity_to_obj(ymap_obj: bpy.types.Object, ymap: CMapData):
    group_obj = bpy.data.objects.new("Entities", None)
    group_obj.sollum_type = SollumType.YMAP_ENTITY_GROUP
    group_obj.parent = ymap_obj
    group_obj.lock_location = (True, True, True)
    group_obj.lock_rotation = (True, True, True)
    group_obj.lock_scale = (True, True, True)
    bpy.context.collection.objects.link(group_obj)
    bpy.context.view_layer.objects.active = group_obj

    found = False
    if ymap.entities:
        for obj in bpy.context.collection.all_objects:
            for entity in ymap.entities:
                if entity.archetype_name == obj.name and obj.name in bpy.context.view_layer.objects:
                    found = True
                    apply_entity_properties(obj, entity)
        if found:
            logger.info(f"Succesfully imported: {ymap.name}.ymap")
            return True
        else:
            logger.info(
                f"No entities from '{ymap.name}.ymap' exist in the view layer!")
            return False
    else:
        logger.error(f"{ymap.name}.ymap contains no entities to import!")
        return False


def instanced_entity_to_obj(ymap_obj: bpy.types.Object, ymap: CMapData):
    group_obj = bpy.data.objects.new("Entities", None)
    group_obj.sollum_type = SollumType.YMAP_ENTITY_GROUP
    group_obj.parent = ymap_obj
    group_obj.lock_location = (True, True, True)
    group_obj.lock_rotation = (True, True, True)
    group_obj.lock_scale = (True, True, True)
    bpy.context.collection.objects.link(group_obj)
    bpy.context.view_layer.objects.active = group_obj

    if ymap.entities:
        entities_amount = len(ymap.entities)
        count = 0

        for entity in ymap.entities:
            obj = bpy.data.objects.get(entity.archetype_name, None)
            if obj is None:
                # No object with the given archetype name found
                continue

            # TODO: requiring ymap entities to be drawable or fragment in blender seems like an unnecessary limitation
            # Need to special case assets because their type when imported by sollumz is drawable model
            if obj.sollum_type == SollumType.DRAWABLE or obj.sollum_type == SollumType.FRAGMENT or obj.asset_data is not None:
                new_obj = duplicate_object_with_children(obj)
                apply_entity_properties(new_obj, entity)
                new_obj.parent = group_obj
                count += 1
                entity.found = True
            else:
                logger.error(
                    f"Cannot use your '{obj.name}' object because it is not a 'Drawable' type!")

        # Creating empty entity if no object was found for reference, and notify user
        import_settings = get_import_settings()

        if not import_settings.ymap_skip_missing_entities:
            for entity in ymap.entities:
                if entity.found is None:
                    empty_obj = bpy.data.objects.new(
                        entity.archetype_name + " (not found)", None)
                    empty_obj.parent = group_obj
                    apply_entity_properties(empty_obj, entity)
                    empty_obj.sollum_type = SollumType.DRAWABLE
                    logger.error(
                        f"'{entity.archetype_name}' is missing in scene, creating an empty drawable instead.")
        if count > 0:
            logger.info(
                f"Succesfully placed {count}/{entities_amount} entities from scene!")
            return group_obj
        else:
            logger.info(
                f"No entity from '{ymap_obj.name}.ymap' exist in the view layer!")
            return False
    else:
        logger.error(f"{ymap_obj.name}.ymap doesn't contains any entity!")
        return False


def box_to_obj(obj, ymap: CMapData):
    group_obj = create_empty_object(SollumType.YMAP_BOX_OCCLUDER_GROUP, "Box Occluders")
    group_obj.parent = obj
    group_obj.lock_location = (True, True, True)
    group_obj.lock_rotation = (True, True, True)
    group_obj.lock_scale = (True, True, True)
    bpy.context.view_layer.objects.active = group_obj

    obj.ymap_properties.content_flags_toggle.has_occl = True

    for box in ymap.box_occluders:
        box_obj = create_blender_object(SollumType.YMAP_BOX_OCCLUDER, "Box")
        box_obj.active_material = add_occluder_material(SollumType.YMAP_BOX_OCCLUDER)
        create_box(box_obj.data, 1)
        box_obj.location = Vector([box.center_x, box.center_y, box.center_z]) / 4
        box_obj.rotation_euler[2] = math.atan2(box.cos_z, box.sin_z)
        box_obj.scale = Vector([box.length, box.width, box.height]) / 4
        box_obj.parent = group_obj

    return group_obj


def model_to_obj(obj: bpy.types.Object, ymap: CMapData):
    group_obj = create_empty_object(SollumType.YMAP_MODEL_OCCLUDER_GROUP, "Model Occluders")
    group_obj.parent = obj
    group_obj.lock_location = (True, True, True)
    group_obj.lock_rotation = (True, True, True)
    group_obj.lock_scale = (True, True, True)
    bpy.context.view_layer.objects.active = group_obj

    obj.ymap_properties.content_flags_toggle.has_occl = True

    for model in ymap.occlude_models:
        verts, faces = occlude_model_to_mesh_data(model)

        mesh = bpy.data.meshes.new("Model Occluders")
        model_obj = create_blender_object(SollumType.YMAP_MODEL_OCCLUDER, "Model", mesh)
        model_obj.ymap_model_occl_properties.model_occl_flags = model.flags
        model_obj.active_material = add_occluder_material(SollumType.YMAP_MODEL_OCCLUDER)
        mesh.from_pydata(verts, [], faces)
        model_obj.parent = group_obj
        model_obj.lock_location = (True, True, True)
        model_obj.lock_rotation = (True, True, True)
        model_obj.lock_scale = (True, True, True)


def cargen_to_obj(obj: bpy.types.Object, ymap: CMapData):
    group_obj = bpy.data.objects.new("Car Generators", None)
    group_obj.sollum_type = SollumType.YMAP_CAR_GENERATOR_GROUP
    group_obj.parent = obj
    group_obj.lock_location = (True, True, True)
    group_obj.lock_rotation = (True, True, True)
    group_obj.lock_scale = (True, True, True)
    bpy.context.collection.objects.link(group_obj)
    bpy.context.view_layer.objects.active = group_obj

    cargen_ref_mesh = get_cargen_mesh()

    for cargen in ymap.car_generators:
        cargen_obj = bpy.data.objects.new("Car Generator", object_data=cargen_ref_mesh)
        cargen_obj.ymap_cargen_properties.orient_x = cargen.orient_x
        cargen_obj.ymap_cargen_properties.orient_y = cargen.orient_y
        cargen_obj.ymap_cargen_properties.perpendicular_length = cargen.perpendicular_length
        cargen_obj.ymap_cargen_properties.car_model = cargen.car_model
        cargen_obj.ymap_cargen_properties.flags = cargen.flags
        cargen_obj.ymap_cargen_properties.body_color_remap_1 = cargen.body_color_remap_1
        cargen_obj.ymap_cargen_properties.body_color_remap_2 = cargen.body_color_remap_2
        cargen_obj.ymap_cargen_properties.body_color_remap_3 = cargen.body_color_remap_3
        cargen_obj.ymap_cargen_properties.body_color_remap_4 = cargen.body_color_remap_4
        cargen_obj.ymap_cargen_properties.pop_group = cargen.pop_group
        cargen_obj.ymap_cargen_properties.livery = cargen.livery

        angl = math.atan2(cargen.orient_x, cargen.orient_y)
        cargen_obj.rotation_euler = Euler((0.0, 0.0, angl * -1))

        cargen_obj.location = cargen.position
        cargen_obj.sollum_type = SollumType.YMAP_CAR_GENERATOR
        cargen_obj.parent = group_obj


def get_grass_instance_mesh() -> bpy.types.Mesh:
    """Get or create a grass instance visualization mesh.
    Creates 8 standing vertical planes randomly positioned within a 0.5 radius area.
    """
    import random
    mesh_name = ".sollumz.grass_instance_mesh"
    mesh = bpy.data.meshes.get(mesh_name, None)
    if mesh is None:
        mesh = bpy.data.meshes.new(mesh_name)
        
        # Use fixed seed for consistent grass pattern
        random.seed(42)
        
        verts = []
        faces = []
        
        # Create 8 standing planes in a 0.5 radius area
        for i in range(8):
            # Random position within 0.5 radius
            angle = random.uniform(0, 2 * 3.14159)
            dist = random.uniform(0, 0.5)
            cx = dist * math.cos(angle)
            cy = dist * math.sin(angle)
            
            # Random rotation for each grass blade
            blade_angle = random.uniform(0, 3.14159)
            
            # Grass blade size (width and height)
            half_width = 0.08
            height = 0.4 + random.uniform(-0.1, 0.1)
            
            # Create vertical plane vertices (standing up)
            dx = half_width * math.cos(blade_angle)
            dy = half_width * math.sin(blade_angle)
            
            base_idx = len(verts)
            verts.extend([
                (cx - dx, cy - dy, 0),       # Bottom left
                (cx + dx, cy + dy, 0),       # Bottom right
                (cx + dx, cy + dy, height),  # Top right
                (cx - dx, cy - dy, height),  # Top left
            ])
            faces.append((base_idx, base_idx + 1, base_idx + 2, base_idx + 3))
        
        mesh.from_pydata(verts, [], faces)
        mesh.update()
    return mesh


def grass_to_obj(obj: bpy.types.Object, ymap: CMapData):
    """Import grass instances from ymap"""
    # Create the Grass group
    group_obj = bpy.data.objects.new("Grass", None)
    group_obj.sollum_type = SollumType.YMAP_GRASS_GROUP
    group_obj.parent = obj
    group_obj.lock_location = (True, True, True)
    group_obj.lock_rotation = (True, True, True)
    group_obj.lock_scale = (True, True, True)
    bpy.context.collection.objects.link(group_obj)
    
    obj.ymap_properties.content_flags_toggle.has_grass = True
    
    grass_mesh = get_grass_instance_mesh()
    
    grass_list = ymap.instanced_data.grass_instance_list
    
    for batch_idx, grass_batch in enumerate(grass_list):
        # Create batch object (empty with properties)
        batch_name = grass_batch.archetype_name if grass_batch.archetype_name else f"Grass Batch {batch_idx}"
        batch_obj = bpy.data.objects.new(batch_name, None)
        batch_obj.empty_display_size = 1.0
        batch_obj.empty_display_type = 'CUBE'
        batch_obj.sollum_type = SollumType.YMAP_GRASS_BATCH
        batch_obj.parent = group_obj
        bpy.context.collection.objects.link(batch_obj)
        
        # Set batch properties
        batch_obj.ymap_grass_batch_properties.archetype_name = grass_batch.archetype_name or "proc_grasses01"
        batch_obj.ymap_grass_batch_properties.lod_dist = grass_batch.lod_dist
        batch_obj.ymap_grass_batch_properties.lod_fade_start_dist = grass_batch.lod_fade_start_dist
        batch_obj.ymap_grass_batch_properties.lod_inst_fade_range = grass_batch.lod_inst_fade_range
        batch_obj.ymap_grass_batch_properties.orient_to_terrain = grass_batch.orient_to_terrain
        
        # Scale range
        if hasattr(grass_batch, 'scale_range') and grass_batch.scale_range is not None:
            batch_obj.ymap_grass_batch_properties.scale_range = (
                grass_batch.scale_range.x,
                grass_batch.scale_range.y,
                grass_batch.scale_range.z
            )
        
        # Get bounding box for position decoding
        bb_min = grass_batch.batch_aabb.min
        bb_max = grass_batch.batch_aabb.max
        
        # Position the batch at the center of its bounding box
        batch_center = Vector((
            (bb_min.x + bb_max.x) / 2,
            (bb_min.y + bb_max.y) / 2,
            (bb_min.z + bb_max.z) / 2
        ))
        batch_obj.location = batch_center
        
        # Process individual grass instances
        if grass_batch.instance_list:
            for inst_idx, instance in enumerate(grass_batch.instance_list):
                # Decode position from packed values (relative to BatchAABB)
                # Position values are 0-65535 representing range from min to max
                pos_values = instance.position  # This should be a list of 3 integers
                
                if isinstance(pos_values, str):
                    # Parse the space-separated position string
                    pos_parts = pos_values.split()
                    pos_x = int(pos_parts[0]) if len(pos_parts) > 0 else 0
                    pos_y = int(pos_parts[1]) if len(pos_parts) > 1 else 0
                    pos_z = int(pos_parts[2]) if len(pos_parts) > 2 else 0
                else:
                    pos_x, pos_y, pos_z = pos_values[0], pos_values[1], pos_values[2]
                
                # Decode to world position
                world_x = bb_min.x + (pos_x / 65535.0) * (bb_max.x - bb_min.x)
                world_y = bb_min.y + (pos_y / 65535.0) * (bb_max.y - bb_min.y)
                world_z = bb_min.z + (pos_z / 65535.0) * (bb_max.z - bb_min.z)
                
                # Create grass instance object
                grass_obj = bpy.data.objects.new(f"Grass Instance", grass_mesh)
                grass_obj.sollum_type = SollumType.YMAP_GRASS_INSTANCE
                grass_obj.location = Vector((world_x, world_y, world_z))
                grass_obj.parent = batch_obj
                bpy.context.collection.objects.link(grass_obj)
                
                # Parse and set color
                color_values = instance.color
                if isinstance(color_values, str):
                    color_parts = color_values.split()
                    r = int(color_parts[0]) / 255.0 if len(color_parts) > 0 else 1.0
                    g = int(color_parts[1]) / 255.0 if len(color_parts) > 1 else 0.0
                    b = int(color_parts[2]) / 255.0 if len(color_parts) > 2 else 0.0
                else:
                    r = color_values[0] / 255.0
                    g = color_values[1] / 255.0
                    b = color_values[2] / 255.0
                
                grass_obj.ymap_grass_instance_properties.color = (r, g, b)
                # Set viewport display color
                grass_obj.color = (r, g, b, 1.0)
                
                # Set other instance properties
                scale_val = instance.scale / 255.0
                grass_obj.scale = (scale_val, scale_val, scale_val)
                grass_obj.ymap_grass_instance_properties.ao = instance.ao / 255.0
                
                # Apply normals to rotation
                # Decode normal X/Y (0-255 to -1.0-1.0)
                nx = (instance.normal_x - 127.0) / 127.0
                ny = (instance.normal_y - 127.0) / 127.0
                
                # Calculate Z component (assuming unit vector length 1)
                # z = sqrt(1 - x^2 - y^2)
                nz_sq = 1.0 - (nx * nx) - (ny * ny)
                nz = math.sqrt(nz_sq) if nz_sq > 0 else 0.0
                
                # Create normal vector and apply to rotation
                normal_vec = Vector((nx, ny, nz))
                grass_obj.rotation_euler = normal_vec.to_track_quat('Z', 'Y').to_euler()


def ymap_to_obj(ymap: CMapData):
    ymap_obj = bpy.data.objects.new(ymap.name, None)
    ymap_obj.sollum_type = SollumType.YMAP
    ymap_obj.lock_location = (True, True, True)
    ymap_obj.lock_rotation = (True, True, True)
    ymap_obj.lock_scale = (True, True, True)
    bpy.context.collection.objects.link(ymap_obj)
    bpy.context.view_layer.objects.active = ymap_obj

    ymap_obj.ymap_properties.parent = ymap.parent
    ymap_obj.ymap_properties.flags = ymap.flags
    ymap_obj.ymap_properties.content_flags = ymap.content_flags

    ymap_obj.ymap_properties.streaming_extents_min = ymap.streaming_extents_min
    ymap_obj.ymap_properties.streaming_extents_max = ymap.streaming_extents_max
    ymap_obj.ymap_properties.entities_extents_min = ymap.entities_extents_min
    ymap_obj.ymap_properties.entities_extents_max = ymap.entities_extents_max

    import_settings = get_import_settings()

    # Entities
    # TODO: find a way to retrieve ignored stuff on export
    if not import_settings.ymap_exclude_entities and ymap.entities:
        if import_settings.ymap_instance_entities:
            instanced_entity_to_obj(ymap_obj, ymap)
        else:
            entity_to_obj(ymap_obj, ymap)

    # Box occluders
    if import_settings.ymap_box_occluders == False and len(ymap.box_occluders) > 0:
        box_to_obj(ymap_obj, ymap)

    # Model occluders
    if import_settings.ymap_model_occluders == False and len(ymap.occlude_models) > 0:
        model_to_obj(ymap_obj, ymap)

    # TODO: physics_dictionaries

    # TODO: time cycle

    # Car generators
    if import_settings.ymap_car_generators == False and len(ymap.car_generators) > 0:
        cargen_to_obj(ymap_obj, ymap)

    # Grass instances
    # instanced_data is always created, check if grass_instance_list has items
    try:
        grass_list = ymap.instanced_data.grass_instance_list
        if grass_list and len(grass_list) > 0:
            grass_to_obj(ymap_obj, ymap)
    except (AttributeError, TypeError):
        pass  # No grass data

    # TODO: lod ligths

    # TODO: distant lod lights

    ymap_obj.ymap_properties.block.version = str(ymap.block.version)
    ymap_obj.ymap_properties.block.flags = str(ymap.block.flags)
    ymap_obj.ymap_properties.block.name = ymap.block.name
    ymap_obj.ymap_properties.block.exported_by = ymap.block.exported_by
    ymap_obj.ymap_properties.block.owner = ymap.block.owner
    ymap_obj.ymap_properties.block.time = ymap.block.time

    # Set ymap obj hierarchy in the active collection
    set_object_collection(ymap_obj)

    return ymap_obj


def import_ymap(filepath):
    ymap_xml: CMapData = YMAP.from_xml_file(filepath)
    found = False
    for obj in bpy.context.scene.objects:
        if obj.sollum_type == SollumType.YMAP and obj.name == ymap_xml.name:
            logger.error(
                f"{ymap_xml.name} is already existing in the scene. Aborting.")
            found = True
            break
    if not found:
        obj = ymap_to_obj(ymap_xml)
