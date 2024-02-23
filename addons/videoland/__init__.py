import bpy
import videoland

bl_info = {
    "name": "Videoland Renderer",
    "description": "A render engine.",
    "author": "weqqr",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "warning": "Doesn't work.",
    "category": "Render",
}


class VideolandRenderer(bpy.types.RenderEngine):
    bl_idname = "VIDEOLAND"
    bl_label = "Videoland"
    bl_use_preview = True

    def __init__(self):
        super().__init__()

        self.renderer = videoland.Renderer()


def register():
    bpy.utils.register_class(VideolandRenderer)


def unregister():
    bpy.utils.unregister_class(VideolandRenderer)
