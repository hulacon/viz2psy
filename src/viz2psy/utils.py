"""Shared utilities: image loading."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .exceptions import ImageLoadError


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB PIL Image.

    Raises
    ------
    ImageLoadError
        If the file is missing, unreadable, or not a valid image.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(path, "file not found")
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        raise ImageLoadError(path, "not a valid image file")
    except PermissionError:
        raise ImageLoadError(path, "permission denied")
    except OSError as e:
        raise ImageLoadError(path, str(e)) from e
