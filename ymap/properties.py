import bpy

from ..tools.utils import int_to_bool_list, flag_prop_to_list, flag_list_to_int
from bpy.props import (EnumProperty, FloatProperty, PointerProperty,
                       StringProperty, IntProperty, FloatVectorProperty, BoolProperty)


def update_uint_prop(self, context, var_name):
    """Work around for storing uint values as a Blender property (credits: CFox)"""
    try:
        int(getattr(self, var_name))
    except ValueError:
        setattr(self, var_name, '0')

    value = int(getattr(self, var_name))
    max_val = (2**32) - 1

    if value < 0:
        setattr(self, var_name, '0')
    elif value > max_val:
        setattr(self, var_name, str(max_val))


# TODO: why is FlagPropertyGroup duplicated here?
class FlagPropertyGroup:
    """Credits: Sollumz"""

    def update_flags_total(self, context):
        flags = int_to_bool_list(int(self.flags), size=2)
        for index, flag_name in enumerate(MapFlags.__annotations__):
            self.flags_toggle[flag_name] = flags[index]

    def update_flag(self, context):
        flags = flag_prop_to_list(self.__class__.__annotations__, self)
        obj = context.active_object
        if obj:
            obj.ymap_properties.flags = flag_list_to_int(flags)


class ContentFlagPropertyGroup:
    """Credits: Sollumz"""

    def update_flags_total(self, context):
        flags = int_to_bool_list(int(self.content_flags), size=11)
        for index, flag_name in enumerate(ContentFlags.__annotations__):
            self.content_flags_toggle[flag_name] = flags[index]

    def update_flag(self, context):
        flags = flag_prop_to_list(self.__class__.__annotations__, self)
        obj = context.active_object
        if obj:
            obj.ymap_properties.content_flags = flag_list_to_int(flags)


class MapFlags(bpy.types.PropertyGroup):
    script_loaded: bpy.props.BoolProperty(
        name="Script Loaded", update=FlagPropertyGroup.update_flag)
    has_lod: bpy.props.BoolProperty(
        name="LOD", update=FlagPropertyGroup.update_flag)


class ContentFlags(bpy.types.PropertyGroup):
    has_hd: BoolProperty(
        name="HD", update=ContentFlagPropertyGroup.update_flag)
    has_lod: BoolProperty(
        name="LOD", update=ContentFlagPropertyGroup.update_flag)
    has_slod2: BoolProperty(
        name="SLOD2", update=ContentFlagPropertyGroup.update_flag)
    has_int: BoolProperty(
        name="Interior", update=ContentFlagPropertyGroup.update_flag)
    has_slod: BoolProperty(
        name="SLOD", update=ContentFlagPropertyGroup.update_flag)
    has_occl: BoolProperty(
        name="Occlusion", update=ContentFlagPropertyGroup.update_flag)
    has_physics: BoolProperty(
        name="Physics", update=ContentFlagPropertyGroup.update_flag)
    has_lod_lights: BoolProperty(
        name="Lod Lights", update=ContentFlagPropertyGroup.update_flag)
    has_dis_lod_lights: BoolProperty(
        name="Distant Lod Lights", update=ContentFlagPropertyGroup.update_flag)
    has_critical: BoolProperty(
        name="Critical", update=ContentFlagPropertyGroup.update_flag)
    has_grass: BoolProperty(
        name="Grass", update=ContentFlagPropertyGroup.update_flag)


class YmapBlockProperties(bpy.types.PropertyGroup):
    version: StringProperty(name="Version", default='0', update=lambda self,
                            context: update_uint_prop(self, context, 'version'))
    flags: StringProperty(name="Flags", default='0', update=lambda self,
                          context: update_uint_prop(self, context, 'flags'))
    # TODO: Sync the name input with the object name?
    name: StringProperty(name="Name")
    exported_by: StringProperty(name="Exported By", default="Sollumz")
    owner: StringProperty(name="Owner")
    time: StringProperty(name="Time")


class YmapProperties(bpy.types.PropertyGroup):
    parent: StringProperty(name="Parent", default="")
    parent_ymap: bpy.props.PointerProperty(type=bpy.types.Object, name="Parent Yimap")
    flags: IntProperty(name="Flags", default=0, min=0, max=3,
                       update=FlagPropertyGroup.update_flags_total)
    content_flags: IntProperty(name="Content Flags", default=0, min=0, max=(
        2**11)-1, update=ContentFlagPropertyGroup.update_flags_total)

    streaming_extents_min: FloatVectorProperty(name="Streaming Extents Min", default=(0, 0, 0), size=3, subtype='XYZ')
    streaming_extents_max: FloatVectorProperty(name="Streaming Extents Max", default=(0, 0, 0), size=3, subtype='XYZ')
    entities_extents_min: FloatVectorProperty(name="Entities Extents Min", default=(0, 0, 0), size=3, subtype='XYZ')
    entities_extents_max: FloatVectorProperty(name="Entities Extents Max", default=(0, 0, 0), size=3, subtype='XYZ')

    flags_toggle: PointerProperty(type=MapFlags)
    content_flags_toggle: PointerProperty(type=ContentFlags)

    block: PointerProperty(
        type=YmapBlockProperties)


class YmapModelOccluderProperties(bpy.types.PropertyGroup):
    model_occl_flags: IntProperty(name="Flags", default=0)

    def _set_is_water_only(self, enable: bool):
        if enable:
            self.model_occl_flags |= 1
        else:
            self.model_occl_flags &= ~1

    is_water_only: BoolProperty(
        name="Is Water Only",
        get=lambda s: (s.model_occl_flags & 1) != 0,
        set=_set_is_water_only
    )


class YmapCarGeneratorProperties(bpy.types.PropertyGroup):
    car_model: StringProperty(name="CarModel", default="panto")
    cargen_flags: IntProperty(name="Flags", default=0)
    pop_group: StringProperty(name="PopGroup", default="")
    perpendicular_length: FloatProperty(
        name="PerpendicularLength", default=2.3)
    body_color_remap_1: IntProperty(name="BodyColorRemap1", default=-1)
    body_color_remap_2: IntProperty(name="BodyColorRemap2", default=-1)
    body_color_remap_3: IntProperty(name="BodyColorRemap3", default=-1)
    body_color_remap_4: IntProperty(name="BodyColorRemap4", default=-1)
    livery: IntProperty(name="Livery", default=-1)


class YmapGrassBatchProperties(bpy.types.PropertyGroup):
    """Properties for a grass batch (GrassInstanceList item)"""
    archetype_name: StringProperty(name="Archetype Name", default="proc_grasses01")
    lod_dist: FloatProperty(name="LOD Distance", default=120.0)
    lod_fade_start_dist: FloatProperty(name="LOD Fade Start Dist", default=15.0)
    lod_inst_fade_range: FloatProperty(name="LOD Inst Fade Range", default=0.75)
    orient_to_terrain: FloatProperty(name="Orient To Terrain", default=1.0)
    scale_range: FloatVectorProperty(name="Scale Range", default=(0.3, 0.2, 0.7), size=3)


def update_grass_instance_color(self, context):
    """Sync grass instance color with viewport display color"""
    obj = context.object
    if obj is not None:
        color = self.color
        obj.color = (color[0], color[1], color[2], 1.0)


class YmapGrassInstanceProperties(bpy.types.PropertyGroup):
    """Properties for an individual grass instance"""
    color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        default=(0.4, 0.7, 0.2),  # Green color
        min=0.0, max=1.0,
        size=3,
        description="Grass instance color (RGB)",
        update=update_grass_instance_color
    )
    ao: FloatProperty(name="Ambient Occlusion", default=1.0, min=0.0, max=1.0, description="Ambient occlusion value (0-1)")


class YmapGrassPaintToolProperties(bpy.types.PropertyGroup):
    """Properties for the grass paint tool"""
    brush_radius: FloatProperty(name="Brush Radius", default=2.0, min=0.1, max=50.0)
    density: FloatProperty(name="Density", default=0.5, min=0.1, max=10.0, description="Grass instances per square unit")
    color: FloatVectorProperty(
        name="Grass Color", 
        subtype='COLOR', 
        default=(0.4, 0.7, 0.2), 
        min=0.0, max=1.0, 
        size=3
    )
    target_object: PointerProperty(type=bpy.types.Object, name="Target Object", description="Object to paint on (leave empty to paint on all visible surfaces)")
    color_variance: FloatProperty(name="Color Variance", default=0.2, min=0.0, max=1.0, description="Random variation in grass color (Hue and Brightness)")


def register():
    bpy.types.Object.ymap_properties = PointerProperty(type=YmapProperties)
    bpy.types.Object.ymap_model_occl_properties = PointerProperty(
        type=YmapModelOccluderProperties)
    bpy.types.Object.ymap_cargen_properties = PointerProperty(
        type=YmapCarGeneratorProperties)
    bpy.types.Object.ymap_grass_batch_properties = PointerProperty(
        type=YmapGrassBatchProperties)
    bpy.types.Object.ymap_grass_instance_properties = PointerProperty(
        type=YmapGrassInstanceProperties)
    bpy.types.Scene.ymap_grass_paint_tool_properties = PointerProperty(type=YmapGrassPaintToolProperties)


def unregister():
    del bpy.types.Object.ymap_properties
    del bpy.types.Object.ymap_model_occl_properties
    del bpy.types.Object.ymap_cargen_properties
    del bpy.types.Object.ymap_grass_batch_properties
    del bpy.types.Object.ymap_grass_instance_properties
    del bpy.types.Scene.ymap_grass_paint_tool_properties

