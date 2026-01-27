import bpy
import math
import random
import bmesh
import colorsys
import traceback
from mathutils import Vector
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils

from ..sollumz_helper import SOLLUMZ_OT_base
from ..sollumz_properties import SollumType


class SOLLUMZ_OT_create_grass_batch(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a new grass batch"""
    bl_idname = "sollumz.create_grass_batch"
    bl_label = "Add Grass Batch"
    bl_description = "Create a new grass batch (group of grass instances with same archetype)"

    def run(self, context):
        group_obj = context.active_object
        
        # Create mesh for grass batch
        mesh = bpy.data.meshes.new("Grass Batch")
        batch_obj = bpy.data.objects.new("Grass Batch", mesh)
        batch_obj.sollum_type = SollumType.YMAP_GRASS_BATCH
        bpy.context.collection.objects.link(batch_obj)
        batch_obj.parent = group_obj
        
        # Select the new object
        bpy.ops.object.select_all(action='DESELECT')
        batch_obj.select_set(True)
        context.view_layer.objects.active = batch_obj
        
        return True


GRASS_INSTANCE_MESH_NAME = ".sollumz.grass_instance_mesh"


def get_grass_instance_mesh() -> bpy.types.Mesh:
    """Get or create a grass instance visualization mesh.
    Creates 8 standing vertical planes randomly positioned within a 0.5 radius area.
    """
    mesh = bpy.data.meshes.get(GRASS_INSTANCE_MESH_NAME, None)
    if mesh is None:
        mesh = bpy.data.meshes.new(GRASS_INSTANCE_MESH_NAME)
        
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


def setup_grass_geometry_nodes(batch_obj):
    """Setup geometry nodes modifier for grass visualization on a batch object.
    Can be called from paint operator or import.
    """
    # Check if geometry nodes modifier exists
    mod = None
    for modifier in batch_obj.modifiers:
        if modifier.type == 'NODES' and modifier.name == "GrassInstances":
            mod = modifier
            break
    
    if mod is None:
        # Create geometry nodes modifier
        mod = batch_obj.modifiers.new(name="GrassInstances", type='NODES')
        
        # Create new node group
        node_group = bpy.data.node_groups.new(name="Grass Instances", type='GeometryNodeTree')
        mod.node_group = node_group
        
        nodes = node_group.nodes
        links = node_group.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create Input/Output nodes
        input_node = nodes.new('NodeGroupInput')
        output_node = nodes.new('NodeGroupOutput')
        input_node.location = (-1000, 0)
        output_node.location = (1000, 0)
        
        # Create sockets
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        
        # --- Create simple 4-vertex plane (quad) ---
        plane_node = nodes.new('GeometryNodeMeshGrid')
        plane_node.location = (-800, -300)
        plane_node.inputs['Size X'].default_value = 0.08
        plane_node.inputs['Size Y'].default_value = 0.4
        plane_node.inputs['Vertices X'].default_value = 2  # Just 2 vertices = 4 total (quad)
        plane_node.inputs['Vertices Y'].default_value = 2
        
        # Rotate plane to be vertical
        rotate_plane = nodes.new('GeometryNodeTransform')
        rotate_plane.location = (-600, -300)
        rotate_plane.inputs['Rotation'].default_value = (1.5708, 0, 0)  # 90 degrees around X
        links.new(plane_node.outputs['Mesh'], rotate_plane.inputs['Geometry'])
        
        # --- Duplicate each vertex 8 times ---
        duplicate_node = nodes.new('GeometryNodeDuplicateElements')
        duplicate_node.location = (-600, 0)
        duplicate_node.domain = 'POINT'
        duplicate_node.inputs['Amount'].default_value = 8
        links.new(input_node.outputs[0], duplicate_node.inputs['Geometry'])
        
        # --- Add random offset (within 0.5 radius) ---
        position_node = nodes.new('GeometryNodeInputPosition')
        position_node.location = (-400, -100)
        
        random_offset = nodes.new('FunctionNodeRandomValue')
        random_offset.location = (-400, -200)
        random_offset.data_type = 'FLOAT_VECTOR'
        random_offset.inputs['Min'].default_value = (-0.5, -0.5, 0)
        random_offset.inputs['Max'].default_value = (0.5, 0.5, 0)
        
        # Seed with index
        index_node = nodes.new('GeometryNodeInputIndex')
        index_node.location = (-600, -250)
        links.new(index_node.outputs[0], random_offset.inputs['Seed'])
        
        # Add offset to position
        add_offset = nodes.new('ShaderNodeVectorMath')
        add_offset.location = (-200, -150)
        add_offset.operation = 'ADD'
        links.new(position_node.outputs[0], add_offset.inputs[0])
        links.new(random_offset.outputs['Value'], add_offset.inputs[1])
        
        # Set position
        set_position = nodes.new('GeometryNodeSetPosition')
        set_position.location = (0, 0)
        links.new(duplicate_node.outputs['Geometry'], set_position.inputs['Geometry'])
        links.new(add_offset.outputs[0], set_position.inputs['Position'])
        
        # --- Instance grass planes on points ---
        instance_node = nodes.new('GeometryNodeInstanceOnPoints')
        instance_node.location = (200, 0)
        links.new(set_position.outputs['Geometry'], instance_node.inputs['Points'])
        links.new(rotate_plane.outputs['Geometry'], instance_node.inputs['Instance'])
        
        # --- Read rotation attribute and combine with random rotation ---
        rotation_attr = nodes.new('GeometryNodeInputNamedAttribute')
        rotation_attr.location = (-200, -350)
        rotation_attr.data_type = 'FLOAT_VECTOR'
        rotation_attr.inputs['Name'].default_value = "grass_rotation"
        
        # Random rotation for variety
        random_rot = nodes.new('FunctionNodeRandomValue')
        random_rot.location = (-200, -450)
        random_rot.data_type = 'FLOAT_VECTOR'
        random_rot.inputs['Min'].default_value = (0, 0, 0)
        random_rot.inputs['Max'].default_value = (0, 0, 6.28318)
        links.new(index_node.outputs[0], random_rot.inputs['Seed'])
        
        # Add rotations together
        combine_rotation = nodes.new('ShaderNodeVectorMath')
        combine_rotation.location = (0, -400)
        combine_rotation.operation = 'ADD'
        links.new(rotation_attr.outputs['Attribute'], combine_rotation.inputs[0])
        links.new(random_rot.outputs['Value'], combine_rotation.inputs[1])
        
        # Apply rotation to instances
        links.new(combine_rotation.outputs[0], instance_node.inputs['Rotation'])
        
        # --- Realize instances ---
        realize_node = nodes.new('GeometryNodeRealizeInstances')
        realize_node.location = (400, 0)
        links.new(instance_node.outputs['Instances'], realize_node.inputs['Geometry'])
        
        # --- Read color attribute ---
        color_attr = nodes.new('GeometryNodeInputNamedAttribute')
        color_attr.location = (400, -250)
        color_attr.data_type = 'FLOAT_COLOR'
        color_attr.inputs['Name'].default_value = "grass_color"
        
        # --- Store color as vertex color for material ---
        store_color = nodes.new('GeometryNodeStoreNamedAttribute')
        store_color.location = (600, 0)
        store_color.data_type = 'FLOAT_COLOR'
        store_color.domain = 'CORNER'  # Use CORNER for vertex colors
        store_color.inputs['Name'].default_value = "Color"
        links.new(realize_node.outputs['Geometry'], store_color.inputs['Geometry'])
        links.new(color_attr.outputs['Attribute'], store_color.inputs['Value'])
        
        # --- Set material ---
        set_mat = nodes.new('GeometryNodeSetMaterial')
        set_mat.location = (800, 0)
        links.new(store_color.outputs['Geometry'], set_mat.inputs['Geometry'])
        
        # Create material
        mat = get_or_create_grass_material()
        if mat:
            set_mat.inputs['Material'].default_value = mat
        
        # Link to output
        links.new(set_mat.outputs['Geometry'], output_node.inputs[0])
    
    # Ensure mesh has required attributes
    ensure_grass_mesh_attributes(batch_obj)


def get_or_create_grass_material():
    """Get or create a simple grass material that displays vertex colors"""
    mat_name = ".sollumz.grass_material"
    mat = bpy.data.materials.get(mat_name)
    
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Output node
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (400, 0)
        
        # Principled BSDF
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        
        # Color Attribute node (reads "Color" attribute)
        color_attr = nodes.new('ShaderNodeAttribute')
        color_attr.location = (-300, 0)
        color_attr.attribute_name = "Color"
        color_attr.attribute_type = 'GEOMETRY'
        
        # Connect nodes
        links.new(color_attr.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def ensure_grass_mesh_attributes(batch_obj):
    """Ensure mesh has required attributes for grass"""
    mesh = batch_obj.data
    if mesh is None:
        return
    
    # Ensure attributes exist
    if "grass_color" not in mesh.attributes:
        mesh.attributes.new(name="grass_color", type='FLOAT_COLOR', domain='POINT')
    if "grass_scale" not in mesh.attributes:
        mesh.attributes.new(name="grass_scale", type='FLOAT', domain='POINT')
    if "grass_rotation" not in mesh.attributes:
        mesh.attributes.new(name="grass_rotation", type='FLOAT_VECTOR', domain='POINT')
    if "grass_ao" not in mesh.attributes:
        mesh.attributes.new(name="grass_ao", type='FLOAT', domain='POINT')


class SOLLUMZ_OT_paint_grass(bpy.types.Operator):
    """Paint grass instances using Geometry Nodes (vertices as instances)"""
    bl_idname = "sollumz.paint_grass"
    bl_label = "Paint Grass"
    bl_description = "Paint grass instances. LMB: Paint, Ctrl+LMB: Erase, Scroll: Size, Ctrl+Scroll: Density"
    bl_options = {'REGISTER', 'UNDO'}
    
    _batch_obj = None
    _target_objects = []
    _is_painting = False
    _last_paint_pos = None
    _draw_handler = None
    _mouse_pos = (0, 0)
    _cached_brush_location = None
    _cached_brush_normal = None
    _last_cache_mouse_pos = None
    
    @property
    def settings(self):
        return bpy.context.scene.ymap_grass_paint_tool_properties

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.sollum_type == SollumType.YMAP_GRASS_BATCH
    
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}
        
        self._batch_obj = context.active_object
        
        # Ensure batch has mesh data
        if self._batch_obj.data is None:
            mesh = bpy.data.meshes.new(f"{self._batch_obj.name}_mesh")
            self._batch_obj.data = mesh
        
        # Setup Geometry Nodes if not present
        self._setup_geometry_nodes(context)
        
        # Setup target object mask
        target_obj = self.settings.target_object
        if target_obj:
            self._target_objects = [target_obj]
        else:
            self._target_objects = []
            
        if not self._target_objects:
            self.report({'INFO'}, "Painting on all surfaces")
        else:
            self.report({'INFO'}, f"Restricted painting to {len(self._target_objects)} object(s)")
            
        # Add draw handler
        args = (self, context)
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_3d, args, 'WINDOW', 'POST_VIEW')
        
        context.window_manager.modal_handler_add(self)
        self.update_header(context)
        
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        context.area.tag_redraw()
        self._mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'FINISHED'}
        
        # Brush size adjustment
        if event.type == 'WHEELUPMOUSE':
            if event.ctrl:
                self.settings.density = min(10.0, self.settings.density * 1.2)
            else:
                self.settings.brush_radius = min(50.0, self.settings.brush_radius * 1.1)
            self.update_header(context)
            return {'RUNNING_MODAL'}
        elif event.type == 'WHEELDOWNMOUSE':
            if event.ctrl:
                self.settings.density = max(0.1, self.settings.density * 0.8)
            else:
                self.settings.brush_radius = max(0.1, self.settings.brush_radius * 0.9)
            self.update_header(context)
            return {'RUNNING_MODAL'}
        
        # Paint interaction
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._is_painting = True
                self._last_paint_pos = None
                if event.ctrl:
                    self.erase_grass(context, event)
                else:
                    self.paint_grass(context, event)
            elif event.value == 'RELEASE':
                self._is_painting = False
                self._last_paint_pos = None
            return {'RUNNING_MODAL'}
        
        # Dragging
        if event.type == 'MOUSEMOVE' and self._is_painting:
            if event.ctrl:
                self.erase_grass(context, event)
            else:
                self.paint_grass(context, event)
            return {'RUNNING_MODAL'}
        
        return {'PASS_THROUGH'}
    
    def finish(self, context):
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.header_text_set(None)
    
    def update_header(self, context):
        mask_status = "All" if not self._target_objects else "Target Object"
        context.area.header_text_set(f"Grass Paint | Radius: {self.settings.brush_radius:.1f} | Density: {self.settings.density:.1f} | Mask: {mask_status} | LMB: Paint, Ctrl+LMB: Erase, Scroll: Size, Ctrl+Scroll: Density")

    def _setup_geometry_nodes(self, context):
        """Setup geometry nodes modifier for grass visualization"""
        setup_grass_geometry_nodes(self._batch_obj)
    
    def _sample_surface_color(self, obj, world_location, context):
        """Sample the texture color from the object's material at the hit location using UV coordinates"""
        try:
            # Check if object has mesh and material
            if not obj.data or not isinstance(obj.data, bpy.types.Mesh) or not obj.data.materials:
                return None
            
            mesh = obj.data
            mat = mesh.materials[0] if mesh.materials else None
            
            if not mat or not mat.use_nodes:
                return None
            
            # Get the depsgraph and evaluated object for accurate intersection
            depsgraph = context.evaluated_depsgraph_get()
            
            # Ray cast to get face index and barycentric coordinates
            # We need to cast again to get the face index
            obj_eval = obj.evaluated_get(depsgraph)
            mesh_eval = obj_eval.data
            
            # Transform world location to object local space
            local_location = obj.matrix_world.inverted() @ world_location
            
            # Find closest polygon to the hit point
            closest_face = None
            min_dist = float('inf')
            
            for poly in mesh_eval.polygons:
                # Get polygon center
                center = poly.center
                dist = (center - local_location).length
                if dist < min_dist:
                    min_dist = dist
                    closest_face = poly
            
            if not closest_face or not mesh_eval.uv_layers:
                # No UV data, fall back to material color
                return self._get_material_base_color(mat)
            
            # Get UV coordinates at the hit point
            uv_layer = mesh_eval.uv_layers.active or mesh_eval.uv_layers[0]
            
            # Calculate barycentric coordinates for UV interpolation
            # Get the face's first vertex UV as approximation (could be improved with proper barycentric)
            loop_start = closest_face.loop_start
            loop_total = closest_face.loop_total
            
            # Average UV of the face (simple approximation)
            uv_sum = [0.0, 0.0]
            for i in range(loop_total):
                loop_idx = loop_start + i
                uv = uv_layer.data[loop_idx].uv
                uv_sum[0] += uv[0]
                uv_sum[1] += uv[1]
            
            uv_coord = [uv_sum[0] / loop_total, uv_sum[1] / loop_total]
            
            # Find image texture node in material
            image_node = None
            nodes = mat.node_tree.nodes
            
            # Try to find the image texture connected to the shader
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    image_node = node
                    break
            
            if not image_node or not image_node.image:
                # No texture, use material base color
                return self._get_material_base_color(mat)
            
            # Sample the image texture at UV coordinates
            image = image_node.image
            
            # Get image pixels
            if not image.pixels:
                return self._get_material_base_color(mat)
            
            # Convert UV to pixel coordinates
            width = image.size[0]
            height = image.size[1]
            
            # Wrap UV coordinates (handle tiling)
            u = uv_coord[0] % 1.0
            v = uv_coord[1] % 1.0
            
            # Convert to pixel coordinates
            x = int(u * width) % width
            y = int(v * height) % height
            
            # Get pixel index (images in Blender are stored bottom-to-top)
            pixel_idx = (y * width + x) * 4  # RGBA = 4 channels
            
            # Sample color from image
            pixels = image.pixels[:]  # Get all pixels as a flat array
            
            if pixel_idx + 3 < len(pixels):
                r = pixels[pixel_idx]
                g = pixels[pixel_idx + 1]
                b = pixels[pixel_idx + 2]
                return (r, g, b)
            
            # Fallback
            return self._get_material_base_color(mat)
            
        except Exception as e:
            # Silently fail and use default color
            # import traceback
            # traceback.print_exc()  # For debugging
            return None
    
    def _get_material_base_color(self, mat):
        """Get base color from material node as fallback"""
        if not mat or not mat.use_nodes:
            if mat and hasattr(mat, 'diffuse_color'):
                return (mat.diffuse_color[0], mat.diffuse_color[1], mat.diffuse_color[2])
            return None
        
        nodes = mat.node_tree.nodes
        
        # Find Principled BSDF
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                base_color_input = node.inputs.get('Base Color')
                if base_color_input and hasattr(base_color_input, 'default_value'):
                    color = base_color_input.default_value
                    if len(color) >= 3:
                        return (color[0], color[1], color[2])
        
        return None

    def ray_cast_surface(self, context, origin, vector):
        """Cast ray to find surface"""
        depsgraph = context.evaluated_depsgraph_get()
        result, location, normal, index, obj, matrix = context.scene.ray_cast(depsgraph, origin, vector)
        
        if not result:
            return None, None, None
        
        # Ignore hits on self (grass batch)
        if obj == self._batch_obj:
            # Try again from slightly further
            new_origin = location + vector.normalized() * 0.1
            result, location, normal, index, obj, matrix = context.scene.ray_cast(depsgraph, new_origin, vector)
            if not result:
                return None, None, None
            
        # If masking is enabled, check if obj is in target list
        if self._target_objects and obj not in self._target_objects:
            return None, None, None
            
        return location, normal, obj
    
    def draw_callback_3d(self, op, context):
        """Draw brush circle - optimized with cached ray cast results"""
        
        radius = self.settings.brush_radius
        
        # Only update ray cast if mouse moved significantly
        needs_update = False
        if self._last_cache_mouse_pos is None:
            needs_update = True
        else:
            dx = self._mouse_pos[0] - self._last_cache_mouse_pos[0]
            dy = self._mouse_pos[1] - self._last_cache_mouse_pos[1]
            if dx*dx + dy*dy > 16:
                needs_update = True
        
        if needs_update:
            self._last_cache_mouse_pos = self._mouse_pos
            region = context.region
            rv3d = context.region_data
            
            view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, self._mouse_pos)
            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self._mouse_pos)
            
            location, normal, _ = self.ray_cast_surface(context, ray_origin, view_vector)
            self._cached_brush_location = location
            self._cached_brush_normal = normal
        
        location = self._cached_brush_location
        normal = self._cached_brush_normal
        
        if location and normal:
            # Draw circle aligned to normal
            rot_quat = normal.to_track_quat('Z', 'Y')
            
            vertices = []
            for i in range(33):
                angle = 2 * math.pi * i / 32
                x = math.cos(angle) * radius
                y = math.sin(angle) * radius
                
                v = Vector((x, y, 0.1))
                v.rotate(rot_quat)
                vertices.append(location + v)
            
            try:
                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            except:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": vertices})
            
            gpu.state.line_width_set(2.0)
            shader.bind()
            shader.uniform_float("color", (0.5, 1.0, 0.5, 1.0))
            batch.draw(shader)
            gpu.state.line_width_set(1.0)
    
    def paint_grass(self, context, event):
        """Add grass vertices to mesh"""
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        location, normal, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        
        if not location:
            return
            
        if self._last_paint_pos is not None:
             if (Vector(location) - Vector(self._last_paint_pos)).length < self.settings.brush_radius * 0.3:
                return
        
        self._last_paint_pos = location
        
        # Calculate count
        area = math.pi * self.settings.brush_radius ** 2
        count = max(1, int(area * self.settings.density))
        
        # Pre-compute values
        rot = normal.to_track_quat('Z', 'Y')
        base_color = self.settings.color
        variance = self.settings.color_variance
        
        # Get mesh and use BMesh for efficient editing
        mesh = self._batch_obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        # Get or create attribute layers
        color_layer = bm.verts.layers.float_color.get("grass_color")
        if color_layer is None:
            color_layer = bm.verts.layers.float_color.new("grass_color")
        
        scale_layer = bm.verts.layers.float.get("grass_scale")
        if scale_layer is None:
            scale_layer = bm.verts.layers.float.new("grass_scale")
        
        rotation_layer = bm.verts.layers.float_vector.get("grass_rotation")
        if rotation_layer is None:
            rotation_layer = bm.verts.layers.float_vector.new("grass_rotation")
        
        ao_layer = bm.verts.layers.float.get("grass_ao")
        if ao_layer is None:
            ao_layer = bm.verts.layers.float.new("grass_ao")
        
        # Pre-generate random values
        angles = [random.uniform(0, 2 * math.pi) for _ in range(count)]
        radii = [math.sqrt(random.uniform(0, 1)) * self.settings.brush_radius for _ in range(count)]
        z_rotations = [random.uniform(0, 2 * math.pi) for _ in range(count)]
        scales = [random.uniform(0.8, 1.2) for _ in range(count)]
        h_vars = [random.uniform(-0.05, 0.05) * variance for _ in range(count)]
        v_vars = [random.uniform(-0.2, 0.2) * variance for _ in range(count)]
        s_vars = [random.uniform(-0.1, 0.1) * variance for _ in range(count)]
        
        # Add vertices
        for i in range(count):
            angle = angles[i]
            r = radii[i]
            
            offset = Vector((math.cos(angle) * r, math.sin(angle) * r, 0))
            offset.rotate(rot)
            
            # Sample surface at this offset
            start_pos = location + offset + normal * 2.0
            down_vec = -normal
            
            hit_location, hit_normal, hit_obj = self.ray_cast_surface(context, start_pos, down_vec)
            
            if hit_location:
                # Create vertex at world position
                # Transform to local space of batch object
                local_pos = self._batch_obj.matrix_world.inverted() @ hit_location
                vert = bm.verts.new(local_pos)
                
                # Determine color
                if self.settings.use_surface_color and hit_obj:
                    # Sample surface texture color
                    sampled_color = self._sample_surface_color(hit_obj, hit_location, context)
                    if sampled_color:
                        base_color = sampled_color
                
                # Set attributes with color variation
                h, s, v = colorsys.rgb_to_hsv(base_color[0], base_color[1], base_color[2])
                h = (h + h_vars[i]) % 1.0
                v = max(0.0, min(1.0, v + v_vars[i]))
                s = max(0.0, min(1.0, s + s_vars[i]))
                r_col, g_col, b_col = colorsys.hsv_to_rgb(h, s, v)
                
                vert[color_layer] = (r_col, g_col, b_col, 1.0)
                vert[scale_layer] = scales[i]
                vert[rotation_layer] = (0, 0, z_rotations[i])
                vert[ao_layer] = 1.0
        
        # Update mesh
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    
    def erase_grass(self, context, event):
        """Remove grass vertices within brush radius"""
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        location, _, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        
        if not location:
            return

        # Transform to local space
        sq_radius = self.settings.brush_radius * self.settings.brush_radius
        
        # Use BMesh
        mesh = self._batch_obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        # Find and remove vertices within radius
        verts_to_remove = []
        for vert in bm.verts:
            world_pos = self._batch_obj.matrix_world @ vert.co
            if (world_pos - location).length_squared <= sq_radius:
                verts_to_remove.append(vert)
        
        for vert in verts_to_remove:
            bm.verts.remove(vert)
        
        # Update mesh
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()


class SOLLUMZ_OT_paint_grass_color(bpy.types.Operator):
    """Paint color on grass instances"""
    bl_idname = "sollumz.paint_grass_color"
    bl_label = "Paint Grass Color"
    bl_description = "Paint color on grass instances. LMB: Paint, Scroll: Size"
    bl_options = {'REGISTER', 'UNDO'}
    
    _batch_obj = None
    _target_objects = []
    _is_painting = False
    _last_paint_pos = None
    _draw_handler = None
    _mouse_pos = (0, 0)
    _cached_brush_location = None
    _cached_brush_normal = None
    _last_cache_mouse_pos = None
    
    @property
    def settings(self):
        return bpy.context.scene.ymap_grass_paint_tool_properties

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.sollum_type == SollumType.YMAP_GRASS_BATCH
    
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}
        
        self._batch_obj = context.active_object
        
        target_obj = self.settings.target_object
        if target_obj:
            self._target_objects = [target_obj]
        else:
            self._target_objects = [obj for obj in context.selected_objects if obj != self._batch_obj and obj.type == 'MESH']
            
        # Add draw handler
        args = (self, context)
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_3d, args, 'WINDOW', 'POST_VIEW')
        
        context.window_manager.modal_handler_add(self)
        self.update_header(context)
        
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        context.area.tag_redraw()
        self._mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'FINISHED'}
        
        if event.type == 'WHEELUPMOUSE':
            self.settings.brush_radius = min(50.0, self.settings.brush_radius * 1.1)
            self.update_header(context)
            return {'RUNNING_MODAL'}
        elif event.type == 'WHEELDOWNMOUSE':
            self.settings.brush_radius = max(0.1, self.settings.brush_radius * 0.9)
            self.update_header(context)
            return {'RUNNING_MODAL'}
        
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._is_painting = True
                self._last_paint_pos = None
                self.paint_grass(context, event)
            elif event.value == 'RELEASE':
                self._is_painting = False
                self._last_paint_pos = None
            return {'RUNNING_MODAL'}
        
        if event.type == 'MOUSEMOVE' and self._is_painting:
            self.paint_grass(context, event)
            return {'RUNNING_MODAL'}
        
        return {'PASS_THROUGH'}
    
    def finish(self, context):
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.area.header_text_set(None)
    
    def update_header(self, context):
        mask_status = "All" if not self._target_objects else "Target Object"
        context.area.header_text_set(f"Grass Color Paint | Radius: {self.settings.brush_radius:.1f} | Color Var: {self.settings.color_variance:.2f} | Mask: {mask_status} | LMB: Paint Color, Scroll: Size")

    def ray_cast_surface(self, context, origin, vector):
        depsgraph = context.evaluated_depsgraph_get()
        curr_origin = origin
        
        # Simplified raycast that just finds location on the painted batch/mesh or scene
        result, location, normal, index, obj, matrix = context.scene.ray_cast(depsgraph, curr_origin, vector)
        if result:
            return location, normal, obj
            
        return None, None, None

    def draw_callback_3d(self, op, context):
        # reuse paint_grass logic or simple circle
        radius = self.settings.brush_radius
        
        needs_update = False
        if self._last_cache_mouse_pos is None:
            needs_update = True
        else:
            dx = self._mouse_pos[0] - self._last_cache_mouse_pos[0]
            dy = self._mouse_pos[1] - self._last_cache_mouse_pos[1]
            if dx*dx + dy*dy > 16:
                needs_update = True
        
        if needs_update:
            self._last_cache_mouse_pos = self._mouse_pos
            region = context.region
            rv3d = context.region_data
            
            view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, self._mouse_pos)
            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self._mouse_pos)
            
            location, normal, _ = self.ray_cast_surface(context, ray_origin, view_vector)
            self._cached_brush_location = location
            self._cached_brush_normal = normal
        
        location = self._cached_brush_location
        normal = self._cached_brush_normal
        
        if location and normal:
            rot_quat = normal.to_track_quat('Z', 'Y')
            vertices = []
            for i in range(33):
                angle = 2 * math.pi * i / 32
                x = math.cos(angle) * radius
                y = math.sin(angle) * radius
                v = Vector((x, y, 0.1))
                v.rotate(rot_quat)
                vertices.append(location + v)
            try:
                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            except:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": vertices})
            gpu.state.line_width_set(2.0)
            shader.bind()
            shader.uniform_float("color", (0.5, 0.8, 1.0, 1.0))  # Blue for color paint
            batch.draw(shader)
            gpu.state.line_width_set(1.0)
            
    def paint_grass(self, context, event):
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        location, _, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        if not location:
            return 
            
        sq_radius = self.settings.brush_radius * self.settings.brush_radius
        target_color = self.settings.color
        variance = self.settings.color_variance
        
        # Use BMesh to modify vertex colors
        mesh = self._batch_obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        # Get color layer
        color_layer = bm.verts.layers.float_color.get("grass_color")
        if not color_layer:
            # Nothing to paint if no color layer exists
            bm.free()
            return
            
        modified = False
        
        # Find vertices within radius
        for vert in bm.verts:
            world_pos = self._batch_obj.matrix_world @ vert.co
            if (world_pos - location).length_squared <= sq_radius:
                # Update color with variance
                h, s, v = colorsys.rgb_to_hsv(target_color[0], target_color[1], target_color[2])
                h_var = random.uniform(-0.05, 0.05) * variance
                v_var = random.uniform(-0.2, 0.2) * variance
                s_var = random.uniform(-0.1, 0.1) * variance
                h = (h + h_var) % 1.0
                v = max(0.0, min(1.0, v + v_var))
                s = max(0.0, min(1.0, s + s_var))
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                
                vert[color_layer] = (r, g, b, 1.0)
                modified = True
        
        if modified:
            bm.to_mesh(mesh)
            mesh.update()
        
        bm.free()
