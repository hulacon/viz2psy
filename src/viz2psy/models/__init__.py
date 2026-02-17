"""Model wrappers for psychological feature extraction.

Imports are lazy to avoid hard failures when optional dependencies
(e.g., resmem, deepgaze-pytorch) are not installed.
"""


def __getattr__(name):
    _registry = {
        "ResMemModel": ".resmem",
        "EmoNetModel": ".emonet",
        "CLIPModel": ".clip",
        "GISTModel": ".gist",
        "LLStatModel": ".llstat",
        "SaliencyModel": ".saliency",
        "DINOv2Model": ".dinov2",
        "AestheticsModel": ".aesthetics",
        "PlacesModel": ".places",
        "YOLOModel": ".yolo",
    }
    if name in _registry:
        import importlib
        module = importlib.import_module(_registry[name], __package__)
        return getattr(module, name)
    if name == "__all__":
        return list(_registry.keys())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
