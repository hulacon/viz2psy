"""Custom exceptions for viz2psy."""


class Viz2PsyError(Exception):
    """Base exception for all viz2psy errors."""


class ImageLoadError(Viz2PsyError):
    """Raised when an image cannot be loaded or decoded."""

    def __init__(self, path, reason=None):
        self.path = path
        msg = f"Failed to load image: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class ModelLoadError(Viz2PsyError):
    """Raised when a model fails to load weights or initialize."""

    def __init__(self, model_name, reason=None):
        self.model_name = model_name
        msg = f"Failed to load model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class DeviceError(Viz2PsyError):
    """Raised when a requested compute device is unavailable or invalid."""

    def __init__(self, device, reason=None):
        self.device = device
        msg = f"Invalid or unavailable device '{device}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class VideoError(Viz2PsyError):
    """Raised when a video file cannot be opened or read."""

    def __init__(self, path, reason=None):
        self.path = path
        msg = f"Failed to process video: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class InferenceError(Viz2PsyError):
    """Raised when model inference fails on one or more images."""

    def __init__(self, model_name, reason=None):
        self.model_name = model_name
        msg = f"Inference failed for model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
