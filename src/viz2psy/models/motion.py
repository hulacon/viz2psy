"""Optical-flow motion statistics — viz2psy's first temporal model.

Video-only: motion is defined between frames, so this model reads the
video itself rather than the extracted frame grid. At each requested
timestamp it takes the native frame there and the NEXT native frame
(one 1/fps step — dense flow assumes small displacements, so flow
between grid frames 0.5 s apart would measure nothing), computes dense
Farnebäck optical flow on grayscale downscaled to ``WORK_HEIGHT``, and
reduces the field to seven statistics:

- ``motion_energy`` — mean flow magnitude
- ``motion_energy_p95`` — 95th-percentile magnitude (localized fast motion)
- ``motion_coherence`` — |mean flow vector| / mean magnitude in [0, 1]:
  ~1 for global translation (camera pan), ~0 for incoherent object
  motion; NaN when there is essentially no motion to be coherent about
- ``motion_horizontal`` / ``motion_vertical`` — signed mean components
  (image coordinates: +x rightward, +y DOWNWARD)
- ``motion_radial`` — mean flow projected on the unit vector away from
  the frame center: positive = expansion (zoom-in / approach / looming),
  negative = contraction
- ``motion_frame_diff`` — mean absolute grayscale pixel difference in
  [0, 1]: the classic visual-change regressor, and the tell for hard
  cuts

Flow-derived values are in frame-heights per second (magnitude / working
height x fps), so they are resolution- and frame-rate-independent.
Across a hard cut the flow field is a large incoherent transient — that
is real visual change, deliberately left in rather than masked;
``motion_frame_diff`` spikes there, and cut-aware handling (e.g.
scene-cut pooling schemas) belongs downstream. Timestamps where no next
frame exists (end of video, read failure) score NaN throughout.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from viz2psy.exceptions import Viz2PsyError, VideoError
from viz2psy.models.base import BaseModel

WORK_HEIGHT = 240  # downscale target for flow computation
COHERENCE_MIN_ENERGY = 1e-3  # frame-heights/s below which coherence is NaN

FEATURE_NAMES = [
    "motion_energy",
    "motion_energy_p95",
    "motion_coherence",
    "motion_horizontal",
    "motion_vertical",
    "motion_radial",
    "motion_frame_diff",
]

_NAN_ROW = {name: float("nan") for name in FEATURE_NAMES}


class MotionModel(BaseModel):
    """Dense optical-flow statistics between native-adjacent frame pairs."""

    name = "motion"
    # One column, one kind (§4.1). The six flow/difference statistics are
    # positional nulls: a timestamp with no NEXT native frame (the video's
    # end; a read failure takes the same path) has no pair to compare.
    # Coherence is also null on a static pair, which is content, so it is
    # declared undefined; its end-of-video NaN sits in a row the undefinable
    # siblings already exclude.
    nulls = {
        **{c: {"means": "undefinable",
               "when": "no next native frame at this timestamp: the video's last grid "
                       "time(s), or (rarely) a frame read failure"}
           for c in FEATURE_NAMES if c != "motion_coherence"},
        "motion_coherence": {"means": "undefined",
                             "when": "mean flow magnitude <= COHERENCE_MIN_ENERGY: no motion "
                                     "has no direction (also NaN on the end-of-video row)"},
    }
    checkpoint = None  # analytic (Farnebäck dense flow), no learned weights
    video_only = True

    def load(self) -> None:
        import cv2

        self.model = cv2

    def predict(self, image: Image.Image) -> dict[str, float]:
        raise Viz2PsyError(
            "motion is a video-only model (flow is defined between frames); "
            "run it on a video input, not on still images"
        )

    def predict_video(
        self,
        video_path: str | Path,
        times: list[float],
        quiet: bool = False,
    ) -> list[dict[str, float]]:
        """Score the video at the given timestamps (one dict per timestamp).

        ``times`` is the caller's frame grid, so the rows align exactly
        with the per-frame models' rows.
        """
        cv2 = self.model
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoError(video_path, "could not open video file")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise VideoError(video_path, "could not determine video frame rate")

            from tqdm import tqdm

            from viz2psy.video import is_interlaced

            iterator = times if quiet else tqdm(times, desc=self.name)
            if is_interlaced(video_path):
                return self._score_interlaced(video_path, [float(t) for t in iterator], fps)
            return [self._score_pair(cap, float(t), fps) for t in iterator]
        finally:
            cap.release()

    def _score_interlaced(self, video_path: Path, times: list[float], fps: float) -> list[dict[str, float]]:
        """Interlaced sources: OpenCV cannot decode them, so read the frame
        pairs deinterlaced through ``viz2psy.video.iter_native_frames``."""
        from viz2psy.video import iter_native_frames

        cv2 = self.model
        starts = [int(t * fps) for t in times]
        needed = set(starts) | {s + 1 for s in starts}
        gray = {
            i: self._prepare(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            for i, rgb in iter_native_frames(video_path, needed)
        }
        return [
            self._score_gray(gray[s], gray[s + 1], fps) if s in gray and s + 1 in gray else dict(_NAN_ROW)
            for s in starts
        ]

    def _score_pair(self, cap, t: float, fps: float) -> dict[str, float]:
        cv2 = self.model
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok_a, frame_a = cap.read()
        ok_b, frame_b = cap.read()  # sequential read: the next native frame
        if not (ok_a and ok_b):
            return dict(_NAN_ROW)
        return self._score_gray(self._prepare(frame_a), self._prepare(frame_b), fps)

    def _score_gray(self, gray_a: np.ndarray, gray_b: np.ndarray, fps: float) -> dict[str, float]:
        cv2 = self.model
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # frame-heights per second, resolution/frame-rate independent
        scale = fps / gray_a.shape[0]
        fx = flow[..., 0] * scale
        fy = flow[..., 1] * scale
        mag = np.hypot(fx, fy)
        energy = float(mag.mean())

        mean_fx = float(fx.mean())
        mean_fy = float(fy.mean())
        coherence = (
            float(np.hypot(mean_fx, mean_fy) / energy)
            if energy > COHERENCE_MIN_ENERGY
            else float("nan")
        )

        h, w = gray_a.shape
        ys, xs = np.mgrid[0:h, 0:w]
        rx = xs - (w - 1) / 2.0
        ry = ys - (h - 1) / 2.0
        r_norm = np.hypot(rx, ry)
        r_norm[r_norm == 0] = 1.0
        radial = float(((fx * rx + fy * ry) / r_norm).mean())

        frame_diff = float(
            np.abs(gray_b.astype(np.float32) - gray_a.astype(np.float32)).mean() / 255.0
        )

        return {
            "motion_energy": energy,
            "motion_energy_p95": float(np.percentile(mag, 95)),
            "motion_coherence": coherence,
            "motion_horizontal": mean_fx,
            "motion_vertical": mean_fy,
            "motion_radial": radial,
            "motion_frame_diff": frame_diff,
        }

    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Grayscale + downscale to WORK_HEIGHT (keep aspect)."""
        cv2 = self.model
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h > WORK_HEIGHT:
            new_w = max(1, round(w * WORK_HEIGHT / h))
            gray = cv2.resize(gray, (new_w, WORK_HEIGHT), interpolation=cv2.INTER_AREA)
        return gray
