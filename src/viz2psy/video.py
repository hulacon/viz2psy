"""Video frame extraction utilities."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
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


#: Leading frames checked for the interlaced flag.
_INTERLACE_PROBE_FRAMES = 8

#: A clip whose decoded frames are mostly a single uniform colour is refused
#: rather than scored. OpenCV's FFmpeg backend returns ``ok=True`` with a
#: blank frame when it cannot convert an interlaced frame ("Cannot convert
#: interlaced to progressive frames"), and every model then scores that one
#: image at every timestamp -- valid output, silently void.
MAX_BLANK_FRAME_FRACTION = 0.5

#: The other face of the same failure: a decoder that hands back one stale,
#: non-uniform picture for every request. A clip of at least
#: ``MIN_FRAMES_FOR_FROZEN_CHECK`` sampled frames in which more than this
#: share are bit-identical to the previous sampled frame is refused. Real
#: footage carries codec noise between samples 0.5 s apart; a clip that
#: genuinely never changes is an image, and should be scored as one.
MAX_FROZEN_FRAME_FRACTION = 0.9
MIN_FRAMES_FOR_FROZEN_CHECK = 10


def is_interlaced(video_path: Path, n_probe: int = _INTERLACE_PROBE_FRAMES) -> bool:
    """True when any of the stream's leading frames is flagged interlaced."""
    import av

    try:
        with av.open(str(video_path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if frame.interlaced_frame:
                    return True
                if i + 1 >= n_probe:
                    break
    except av.error.FFmpegError as e:
        raise VideoError(video_path, f"could not probe for interlacing: {e}") from e
    return False


def iter_native_frames(video_path: Path, indices):
    """Yield ``(index, rgb_uint8_array)`` for the requested native frame indices.

    For interlaced sources, which OpenCV cannot decode (see
    ``MAX_BLANK_FRAME_FRACTION``). Decodes sequentially with PyAV through
    FFmpeg's ``yadif`` deinterlacer in ``send_frame`` mode -- one output frame
    per input frame, so output index ``n`` is native frame ``n``, the same
    index ``cap.set(CAP_PROP_POS_FRAMES, n)`` addresses. Indices are yielded
    in increasing order; indices past the end of the stream are not yielded.
    """
    import av

    wanted = sorted({int(i) for i in indices if int(i) >= 0})
    if not wanted:
        return
    want, last = set(wanted), wanted[-1]
    n = 0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        graph = av.filter.Graph()
        src = graph.add_buffer(template=stream)
        yadif = graph.add("yadif", "mode=send_frame:parity=auto:deint=all")
        sink = graph.add("buffersink")
        src.link_to(yadif)
        yadif.link_to(sink)
        graph.configure()

        def drain():
            nonlocal n
            while True:
                try:
                    out = graph.pull()
                except (av.BlockingIOError, av.EOFError):
                    return
                if n in want:
                    yield n, out.to_ndarray(format="rgb24")
                n += 1

        for frame in container.decode(stream):
            graph.push(frame)
            yield from drain()
            if n > last:
                return
        graph.push(None)  # flush yadif's one-frame lookahead
        yield from drain()


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

    Raises
    ------
    VideoError
        If more than ``MAX_BLANK_FRAME_FRACTION`` of the decoded frames are a
        single uniform colour (a decode failure, not a stimulus).

    Progressive sources are read with OpenCV, unchanged. Interlaced sources
    are deinterlaced and read with PyAV (``iter_native_frames``) at the same
    native frame indices.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoError(video_path, "file not found")

    interlaced = is_interlaced(video_path)
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

        if interlaced:
            decoded = iter_native_frames(video_path, (int(t * fps) for t in timestamps))
            pending = next(decoded, None)

        n_blank = n_frozen = 0
        previous = None
        for t in iterator:
            frame_num = int(t * fps)
            if interlaced:
                while pending is not None and pending[0] < frame_num:
                    pending = next(decoded, None)
                if pending is None or pending[0] != frame_num:
                    break
                frame_rgb = pending[1]
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            n_blank += int(frame_rgb.min() == frame_rgb.max())
            n_frozen += int(previous is not None and np.array_equal(frame_rgb, previous))
            previous = frame_rgb
            pil_image = Image.fromarray(frame_rgb)

            if save_dir:
                ext = "jpg" if frame_format == "jpg" else "png"
                frame_path = save_dir / f"frame_{t:.3f}.{ext}"
                save_kwargs = {"quality": 85} if ext == "jpg" else {}
                pil_image.save(frame_path, **save_kwargs)
                frames.append((t, frame_path))
            else:
                frames.append((t, pil_image))

        if frames and n_blank / len(frames) > MAX_BLANK_FRAME_FRACTION:
            raise VideoError(
                video_path,
                f"{n_blank} of {len(frames)} decoded frames are a single uniform colour; "
                "refusing to score them (a decode failure, e.g. an interlaced source "
                "the decoder could not convert)",
            )
        if (len(frames) >= MIN_FRAMES_FOR_FROZEN_CHECK
                and n_frozen / (len(frames) - 1) > MAX_FROZEN_FRAME_FRACTION):
            raise VideoError(
                video_path,
                f"{n_frozen} of {len(frames) - 1} sampled frames are bit-identical to the "
                "previous one; refusing to score a frozen decode (if the video really "
                "never changes, score it as an image)",
            )
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
