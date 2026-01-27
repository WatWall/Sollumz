import bpy
import math
import random
from ..sollumz_helper import SOLLUMZ_OT_base, set_object_collection
from ..tools.ymaphelper import add_occluder_material, create_ymap, create_ymap_group, get_cargen_mesh, generate_ymap_extents
from ..sollumz_properties import SollumType


class SOLLUMZ_OT_create_ymap(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz YMAP object"""
    bl_idname = "sollumz.createymap"
    bl_label = f"Create YMAP"
    bl_description = "Create a YMAP"

    def run(self, context):
        ymap_obj = create_ymap()
        if ymap_obj:
            bpy.ops.object.select_all(action='DESELECT')
            ymap_obj.select_set(True)
            context.view_layer.objects.active = ymap_obj
        return {"FINISHED"}


class SOLLUMZ_OT_create_entity_group(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Entities' group object"""
    bl_idname = "sollumz.create_entity_group"
    bl_label = f"Entities"
    bl_description = "Create 'Entities' group.\n\nOnly 1 per YMAP maximum.\nYou can't create 'Entities' group if the current YMAP already includes occlusions"

    @classmethod
    def poll(cls, context):
        aobj = context.active_object
        if aobj is None:
            return False

        existing_groups = []
        # Do not let user create Entities Group if there is already one, and if there is any kind of Occlusion Group
        for child in aobj.children:
            existing_groups.append(child.name)
        for group in existing_groups:
            if group == "Entities" or group == "Box Occluders" or group == "Model Occluders":
                return False
        return True

    def run(self, context):
        ymap_obj = context.active_object
        create_ymap_group(sollum_type=SollumType.YMAP_ENTITY_GROUP, selected_ymap=ymap_obj, empty_name='Entities')
        # TODO: Find a way to use "bpy.ops.outliner.show_active()" to show the new object in outliner. But we are in wrong context here.
        return True


class SOLLUMZ_OT_create_model_occluder_group(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Model Occluders' group object"""
    bl_idname = "sollumz.create_model_occluder_group"
    bl_label = f"Model Occluders"
    bl_description = "Create 'Model Occluders' group.\n\nOnly 1 per YMAP maximum.\nYou can't create 'Model Occluders' group if the current YMAP already includes an 'Entities' group"

    @classmethod
    def poll(cls, context):
        aobj = context.active_object
        if aobj is None:
            return False

        existing_groups = []
        # Do not let user create Model Occluders Group if there is already one, and if there is already Entities Group
        for child in aobj.children:
            existing_groups.append(child.name)
        for group in existing_groups:
            if group == "Entities" or group == "Model Occluders":
                return False
        return True

    def run(self, context):
        ymap_obj = context.active_object
        create_ymap_group(sollum_type=SollumType.YMAP_MODEL_OCCLUDER_GROUP, selected_ymap=ymap_obj, empty_name='Model Occluders')
        return True


class SOLLUMZ_OT_create_box_occluder_group(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Box Occluders' group object"""
    bl_idname = "sollumz.create_box_occluder_group"
    bl_label = f"Box Occluders"
    bl_description = "Create 'Box Occluders' group.\n\nOnly 1 per YMAP maximum.\nYou can't create 'Box Occluders' group if the current YMAP already includes an 'Entities' group"

    @classmethod
    def poll(cls, context):
        aobj = context.active_object
        if aobj is None:
            return False

        existing_groups = []
        # Do not let user create Box Occluders Group if there is already one, and if there is already Entities Group
        for child in aobj.children:
            existing_groups.append(child.name)
        for group in existing_groups:
            if group == "Entities" or group == "Box Occluders":
                return False
        return True

    def run(self, context):
        ymap_obj = context.active_object
        create_ymap_group(sollum_type=SollumType.YMAP_BOX_OCCLUDER_GROUP, selected_ymap=ymap_obj, empty_name='Box Occluders')
        return True


class SOLLUMZ_OT_create_car_generator_group(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Car Generators' group object"""
    bl_idname = "sollumz.create_car_generator_group"
    bl_label = f"Car Generators"
    bl_description = "Create 'Car Generators' group.\n\nOnly 1 per YMAP maximum"

    @classmethod
    def poll(cls, context):
        aobj = context.active_object
        if aobj is None:
            return False

        existing_groups = []
        for child in aobj.children:
            existing_groups.append(child.name)
        for group in existing_groups:
            if group == "Car Generators":
                return False
        return True

    def run(self, context):
        ymap_obj = context.active_object
        create_ymap_group(sollum_type=SollumType.YMAP_CAR_GENERATOR_GROUP, selected_ymap=ymap_obj, empty_name='Car Generators')
        return True


class SOLLUMZ_OT_create_box_occluder(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Box Occluder' object"""
    bl_idname = "sollumz.create_box_occluder"
    bl_label = "Create Box Occluder"
    bl_description = "Create a 'Box Occluder' object"

    def run(self, context):
        group_obj = context.active_object
        bpy.ops.mesh.primitive_cube_add(size=2)
        box_obj = bpy.context.view_layer.objects.active
        box_obj.sollum_type = SollumType.YMAP_BOX_OCCLUDER
        box_obj.name = "Box"
        box_obj.active_material = add_occluder_material(SollumType.YMAP_BOX_OCCLUDER)
        box_obj.parent = group_obj

        # Prevent rotation on X and Y axis, since only Z axis is supported on Box Occluders
        box_obj.lock_rotation[0] = True
        box_obj.lock_rotation[1] = True

        return True


class SOLLUMZ_OT_create_model_occluder(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Model Occluder' object"""
    bl_idname = "sollumz.create_model_occluder"
    bl_label = "Create Model Occluder"
    bl_description = "Create a 'Model Occluder' object"

    def run(self, context):
        group_obj = context.active_object
        bpy.ops.mesh.primitive_cube_add(size=1)
        model_obj = bpy.context.view_layer.objects.active
        model_obj.name = "Model"
        model_obj.sollum_type = SollumType.YMAP_MODEL_OCCLUDER
        model_obj.ymap_properties.flags = 0
        model_obj.active_material = add_occluder_material(SollumType.YMAP_MODEL_OCCLUDER)
        set_object_collection(model_obj)
        bpy.context.view_layer.objects.active = model_obj
        model_obj.parent = group_obj

        return True


class SOLLUMZ_OT_create_car_generator(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Car Generator' object"""
    bl_idname = "sollumz.create_car_generator"
    bl_label = "Create Car Generator"
    bl_description = "Create a 'Car Generator' object"

    def run(self, context):
        group_obj = context.active_object
        cargen_ref_mesh = get_cargen_mesh()
        cargen_obj = bpy.data.objects.new("Car Generator", object_data=cargen_ref_mesh)
        cargen_obj.sollum_type = SollumType.YMAP_CAR_GENERATOR
        cargen_obj.ymap_cargen_properties.orient_x = 0.0
        cargen_obj.ymap_cargen_properties.orient_y = 0.0
        cargen_obj.ymap_cargen_properties.perpendicular_length = 2.3
        cargen_obj.ymap_cargen_properties.car_model = ""
        cargen_obj.ymap_cargen_properties.flags = 0
        cargen_obj.ymap_cargen_properties.body_color_remap_1 = -1
        cargen_obj.ymap_cargen_properties.body_color_remap_2 = -1
        cargen_obj.ymap_cargen_properties.body_color_remap_3 = -1
        cargen_obj.ymap_cargen_properties.body_color_remap_4 = -1
        cargen_obj.ymap_cargen_properties.pop_group = ""
        cargen_obj.ymap_cargen_properties.livery = -1
        bpy.context.collection.objects.link(cargen_obj)
        bpy.context.view_layer.objects.active = cargen_obj
        cargen_obj.parent = group_obj

        # Prevent rotation on X and Y axis, since only Z axis is supported on Box Occluders
        cargen_obj.lock_rotation[0] = True
        cargen_obj.lock_rotation[1] = True

        # Select the cargen object
        bpy.ops.object.select_all(action='DESELECT')
        cargen_obj.select_set(True)
        context.view_layer.objects.active = cargen_obj

        return True

class SOLLUMZ_OT_generate_ymap_extents(bpy.types.Operator):
    bl_idname = "sollumz.generate_ymap_extents"
    bl_label = "Generate Ymap Extents"
    bl_description = "Generate the YMAP's streaming and entity extents (using YMAP and YTYP entities data)"

    def execute(self, context):
        generate_ymap_extents(selected_ymap=context.active_object)
        return {"FINISHED"}

class SOLLUMZ_OT_convert_to_ymap_entity(SOLLUMZ_OT_base, bpy.types.Operator):
    """Update all objects under the Entities group of the active YMAP to Ymap Entity type"""
    bl_idname = "sollumz.convert_to_ymap_entity"
    bl_label = "Convert to Ymap Entity"
    bl_description = "Update all children of the 'Entities' group to have the 'Ymap Entity' Sollum Type"

    def run(self, context):
        aobj = context.active_object
        
        # Check if we are selecting YMAP or ENTITY GROUP
        entities_group = None
        
        if aobj.sollum_type == SollumType.YMAP_ENTITY_GROUP:
            entities_group = aobj
        elif aobj.sollum_type == SollumType.YMAP:
            for child in aobj.children:
                if child.sollum_type == SollumType.YMAP_ENTITY_GROUP:
                    entities_group = child
                    break
        
        if entities_group:
            count = 0
            for child in entities_group.children:
                child.sollum_type = SollumType.YMAP_ENTITY
                count += 1
            self.report({'INFO'}, f"Converted {count} objects to Ymap Entity type.")
            return True
        else:
            self.report({'WARNING'}, "No Entities group found.")
            return False


class SOLLUMZ_OT_create_grass_group(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a sollumz 'Grass' group object"""
    bl_idname = "sollumz.create_grass_group"
    bl_label = f"Grass Group"
    bl_description = "Create 'Grass' group.\n\nOnly 1 per YMAP maximum"

    @classmethod
    def poll(cls, context):
        aobj = context.active_object
        if aobj is None:
            return False

        existing_groups = []
        for child in aobj.children:
            existing_groups.append(child.name)
        for group in existing_groups:
            if group == "Grass":
                return False
        return True

    def run(self, context):
        ymap_obj = context.active_object
        grass_group = create_ymap_group(sollum_type=SollumType.YMAP_GRASS_GROUP, selected_ymap=ymap_obj, empty_name='Grass')
        # Set the grass content flag
        ymap_obj.ymap_properties.content_flags_toggle.has_grass = True
        return True


class SOLLUMZ_OT_create_grass_batch(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a new grass batch"""
    bl_idname = "sollumz.create_grass_batch"
    bl_label = "Add Grass Batch"
    bl_description = "Create a new grass batch (group of grass instances with same archetype)"

    def run(self, context):
        group_obj = context.active_object
        batch_obj = bpy.data.objects.new("Grass Batch", None)
        batch_obj.empty_display_size = 1.0
        batch_obj.empty_display_type = 'CUBE'
        batch_obj.sollum_type = SollumType.YMAP_GRASS_BATCH
        bpy.context.collection.objects.link(batch_obj)
        batch_obj.parent = group_obj
        
        # Select the new object
        bpy.ops.object.select_all(action='DESELECT')
        batch_obj.select_set(True)
        context.view_layer.objects.active = batch_obj
        
        return True


class SOLLUMZ_OT_create_grass_instance(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create a new grass instance"""
    bl_idname = "sollumz.create_grass_instance"
    bl_label = "Add Grass Instance"
    bl_description = "Create a new grass instance (individual grass point)"

    def run(self, context):
        batch_obj = context.active_object
        
        # Create a simple plane mesh for visualization
        mesh = get_grass_instance_mesh()
        grass_obj = bpy.data.objects.new("Grass Instance", mesh)
        grass_obj.sollum_type = SollumType.YMAP_GRASS_INSTANCE
        
        # Set default color and apply to viewport display
        grass_obj.ymap_grass_instance_properties.color = (0.4, 0.7, 0.2)
        grass_obj.color = (0.4, 0.7, 0.2, 1.0)
        
        bpy.context.collection.objects.link(grass_obj)
        grass_obj.parent = batch_obj
        
        # Position at cursor or slightly offset from parent
        grass_obj.location = context.scene.cursor.location
        
        # Select the new object
        bpy.ops.object.select_all(action='DESELECT')
        grass_obj.select_set(True)
        context.view_layer.objects.active = grass_obj
        
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


class SOLLUMZ_OT_paint_grass(bpy.types.Operator):
    """Paint grass instances on surfaces"""
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
        
        # Setup target object mask from property or selection
        target_obj = self.settings.target_object
        if target_obj:
            self._target_objects = [target_obj]
        else:
            # Fallback to selection if property is empty, just for convenience (optional)
            self._target_objects = [obj for obj in context.selected_objects if obj != self._batch_obj and obj.type == 'MESH']
            
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

    def ray_cast_surface(self, context, origin, vector):
        """Cast ray ignoring grass instances"""
        depsgraph = context.evaluated_depsgraph_get()
        curr_origin = origin
        
        # Try up to 5 times to penetrate grass (stacked instances)
        for _ in range(5):
            result, location, normal, index, obj, matrix = context.scene.ray_cast(depsgraph, curr_origin, vector)
            if not result:
                return None, None, None
            
            # If we hit grass, continue from slightly deeper
            if obj.sollum_type == SollumType.YMAP_GRASS_INSTANCE:
                curr_origin = location + vector.normalized() * 0.05
                continue
                
            # If masking is enabled, check if obj is in target list
            if self._target_objects and obj not in self._target_objects:
                 curr_origin = location + vector.normalized() * 0.05
                 continue
                 
            return location, normal, obj
            
        return None, None, None

    def draw_callback_3d(self, op, context):
        """Draw brush circle"""
        import gpu
        from gpu_extras.batch import batch_for_shader
        from mathutils import Vector
        
        # Calculate circle points
        vertices = []
        radius = self.settings.brush_radius
        
        # Find visualization position under mouse
        region = context.region
        rv3d = context.region_data
        from bpy_extras import view3d_utils
        
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, self._mouse_pos)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self._mouse_pos)
        
        location, normal, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        
        if location:
            # Draw circle aligned to normal
            rot_quat = normal.to_track_quat('Z', 'Y')
            
            for i in range(33):
                angle = 2 * math.pi * i / 32
                x = math.cos(angle) * radius
                y = math.sin(angle) * radius
                
                # Rotate and translate
                v = Vector((x, y, 0.1)) # Lift slightly
                v.rotate(rot_quat)
                vertices.append(location + v)
            
            try:
                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            except:
                # Fallback for newer Blender versions if shader name changed
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": vertices})
            
            gpu.state.line_width_set(2.0)
            shader.bind()
            shader.uniform_float("color", (0.5, 1.0, 0.5, 1.0))
            batch.draw(shader)
            gpu.state.line_width_set(1.0)
    
    def paint_grass(self, context, event):
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        
        from bpy_extras import view3d_utils
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        location, normal, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        
        if not location:
            return
            
        if self._last_paint_pos is not None:
             from mathutils import Vector
             if (Vector(location) - Vector(self._last_paint_pos)).length < self.settings.brush_radius * 0.3:
                return
        
        self._last_paint_pos = location
        
        # Calculate count
        area = math.pi * self.settings.brush_radius ** 2
        count = max(1, int(area * self.settings.density))
        
        grass_mesh = get_grass_instance_mesh()
        
        from mathutils import Vector, Quaternion
        for _ in range(count):
            # Random local offset
            angle = random.uniform(0, 2 * math.pi)
            r = math.sqrt(random.uniform(0, 1)) * self.settings.brush_radius
            
            # Create tangent basis for circle distribution
            rot = normal.to_track_quat('Z', 'Y')
            offset = Vector((math.cos(angle) * r, math.sin(angle) * r, 0))
            offset.rotate(rot)
            
            start_pos = location + offset + normal * 2.0 # Start above
            down_vec = -normal
            
            # Find surface height at this offset
            hit_pos, hit_norm, _ = self.ray_cast_surface(context, start_pos, down_vec)
            
            if hit_pos:
                # Create instance
                grass_obj = bpy.data.objects.new("Grass Instance", grass_mesh)
                grass_obj.sollum_type = SollumType.YMAP_GRASS_INSTANCE
                
                # Align to world surface normal using matrix_world
                # Use rotation_difference for robust alignment of Z-axis to normal
                import mathutils
                rot_quat = mathutils.Vector((0, 0, 1)).rotation_difference(hit_norm)
                
                # Apply world transformation
                mat_loc = mathutils.Matrix.Translation(hit_pos)
                mat_rot = rot_quat.to_matrix().to_4x4()
                grass_obj.matrix_world = mat_loc @ mat_rot
                
                # Parent the object (preserves world transform)
                grass_obj.parent = self._batch_obj
                grass_obj.matrix_parent_inverse = self._batch_obj.matrix_world.inverted()
                
                # Add random rotation Z (visual variation)
                # We rotate around LOCAL Z to spin the grass without changing its up-vector
                grass_obj.rotation_euler.rotate_axis("Z", random.uniform(0, 2 * math.pi))
                
                # Set Properties
                # Set Properties
                color = self.settings.color
                variance = self.settings.color_variance
                
                # Apply variation
                import colorsys
                # Convert to HSV
                h, s, v = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                
                # Vary Hue slightly (+/- 0.05 * variance)
                h += random.uniform(-0.05, 0.05) * variance
                h = h % 1.0
                
                # Vary Value (Brightness) more (+/- 0.2 * variance)
                v += random.uniform(-0.2, 0.2) * variance
                v = max(0.0, min(1.0, v))
                
                # Vary Saturation slightly (+/- 0.1 * variance)
                s += random.uniform(-0.1, 0.1) * variance
                s = max(0.0, min(1.0, s))
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                
                grass_obj.ymap_grass_instance_properties.color = (r, g, b)
                grass_obj.color = (r, g, b, 1.0)
                
                bpy.context.collection.objects.link(grass_obj)

    def erase_grass(self, context, event):
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        
        from bpy_extras import view3d_utils
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        location, _, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        
        if not location:
            return

        # Find and remove grass
        to_delete = []
        from mathutils import Vector
        center = Vector(location)
        sq_radius = self.settings.brush_radius * self.settings.brush_radius
        
        # Optimization: Only check children of batch
        for child in self._batch_obj.children:
            if child.sollum_type == SollumType.YMAP_GRASS_INSTANCE:
                if (child.location - center).length_squared <= sq_radius:
                    to_delete.append(child)
        
        for obj in to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)


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
        for _ in range(5):
            result, location, normal, index, obj, matrix = context.scene.ray_cast(depsgraph, curr_origin, vector)
            if not result:
                return None, None, None
            if obj.sollum_type == SollumType.YMAP_GRASS_INSTANCE:
                curr_origin = location + vector.normalized() * 0.05
                continue
            if self._target_objects and obj not in self._target_objects:
                 curr_origin = location + vector.normalized() * 0.05
                 continue
            return location, normal, obj
        return None, None, None

    def draw_callback_3d(self, op, context):
        import gpu
        from gpu_extras.batch import batch_for_shader
        from mathutils import Vector
        vertices = []
        radius = self.settings.brush_radius
        region = context.region
        rv3d = context.region_data
        from bpy_extras import view3d_utils
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, self._mouse_pos)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self._mouse_pos)
        location, normal, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        if location:
            rot_quat = normal.to_track_quat('Z', 'Y')
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
            shader.uniform_float("color", (0.5, 0.8, 1.0, 1.0)) # Blue for color paint
            batch.draw(shader)
            gpu.state.line_width_set(1.0)
            
    def paint_grass(self, context, event):
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        from bpy_extras import view3d_utils
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        location, _, _ = self.ray_cast_surface(context, ray_origin, view_vector)
        if not location:
            return 
        from mathutils import Vector
        center = Vector(location)
        sq_radius = self.settings.brush_radius * self.settings.brush_radius
        import random
        import colorsys
        target_color = self.settings.color
        variance = self.settings.color_variance
        for child in self._batch_obj.children:
            if child.sollum_type == SollumType.YMAP_GRASS_INSTANCE:
                if (child.location - center).length_squared <= sq_radius:
                    h, s, v = colorsys.rgb_to_hsv(target_color[0], target_color[1], target_color[2])
                    h_var = random.uniform(-0.05, 0.05) * variance
                    v_var = random.uniform(-0.2, 0.2) * variance
                    s_var = random.uniform(-0.1, 0.1) * variance
                    h = (h + h_var) % 1.0
                    v = max(0.0, min(1.0, v + v_var))
                    s = max(0.0, min(1.0, s + s_var))
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    child.ymap_grass_instance_properties.color = (r, g, b)
                    child.color = (r, g, b, 1.0)
