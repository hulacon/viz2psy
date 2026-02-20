"""Video frame extraction utilities."""

import tempfile
from pathlib import Path

import cv2
import psutil
from PIL import Image
from tqdm import tqdm

from .exceptions import VideoError

# Common video file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def is_video_file(path: Path) -> bool:
    """Check if a path is a video file based on extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_video_info(video_path: Path) -> dict:
    """Get video metadata.

    Returns
    -------
    dict
        Keys: fps, frame_count, duration, width, height
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoError(video_path, "file not found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(video_path, "could not open video file")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise VideoError(video_path, "could not determine video frame rate")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def estimate_memory_usage(video_info: dict, frame_interval: float) -> int:
    """Estimate memory usage in bytes for extracted frames.

    Assumes RGB images (3 bytes per pixel) plus some overhead.
    """
    n_frames = int(video_info["duration"] / frame_interval) + 1
    bytes_per_frame = video_info["width"] * video_info["height"] * 3
    # Add ~50% overhead for PIL objects and processing
    return int(n_frames * bytes_per_frame * 1.5)


def get_available_memory() -> int:
    """Get available system memory in bytes."""
    return psutil.virtual_memory().available


def extract_frames(
    video_path: Path,
    frame_interval: float = 0.5,
    save_dir: Path | None = None,
    quiet: bool = False,
    frame_format: str = "jpg",
) -> list[tuple[float, Image.Image | Path]]:
    """Extract frames from a video at specified time intervals.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    frame_interval : float
        Time between frames in seconds (default: 0.5).
    save_dir : Path, optional
        If provided, save frames to this directory and return paths instead of
        PIL Images. Useful for large videos to avoid memory issues.
    quiet : bool
        Suppress progress output.
    frame_format : str
        Image format for saved frames: ``"jpg"`` (default) or ``"png"``.

    Returns
    -------
    list of (time, frame)
        Each entry is (timestamp_in_seconds, PIL.Image or Path).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoError(video_path, "file not found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(video_path, "could not open video file")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise VideoError(video_path, "could not determine video frame rate")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        timestamps = []
        t = 0.0
        while t <= duration:
            timestamps.append(t)
            t += frame_interval

        iterator = timestamps
        if not quiet:
            iterator = tqdm(timestamps, desc="Extracting frames")

        for t in iterator:
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            if save_dir:
                ext = "jpg" if frame_format == "jpg" else "png"
                frame_path = save_dir / f"frame_{t:.3f}.{ext}"
                save_kwargs = {"quality": 85} if ext == "jpg" else {}
                pil_image.save(frame_path, **save_kwargs)
                frames.append((t, frame_path))
            else:
                frames.append((t, pil_image))

        return frames

    finally:
        cap.release()


def extract_frames_to_temp(
    video_path: Path,
    frame_interval: float = 0.5,
    quiet: bool = False,
    frame_format: str = "jpg",
) -> tuple[list[tuple[float, Path]], tempfile.TemporaryDirectory]:
    """Extract frames to a temporary directory.

    Returns the frames list and the TemporaryDirectory object (caller must
    keep a reference to prevent cleanup).

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    frame_interval : float
        Time between frames in seconds.
    quiet : bool
        Suppress progress output.
    frame_format : str
        Image format for saved frames: ``"jpg"`` (default) or ``"png"``.

    Returns
    -------
    frames : list of (time, Path)
    temp_dir : tempfile.TemporaryDirectory
        Keep a reference to prevent automatic cleanup.
    """
    temp_dir = tempfile.TemporaryDirectory(prefix="viz2psy_frames_")
    frames = extract_frames(
        video_path,
        frame_interval=frame_interval,
        save_dir=Path(temp_dir.name),
        quiet=quiet,
        frame_format=frame_format,
    )
    return frames, temp_dir
