"""Shared utilities: image loading."""

from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB PIL Image."""
    return Image.open(path).convert("RGB")
