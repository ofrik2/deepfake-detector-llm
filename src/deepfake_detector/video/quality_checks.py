from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .reader import open_video, read_video_meta, read_frame_at


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict


def _mean_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _blur_score_laplacian(frame_bgr: np.ndarray) -> float:
    """
    Higher = sharper (heuristic). Very low values mean blurry.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run_quality_checks(
    video_path: str,
    *,
    sample_frames: int = 5,
    min_frame_count: int = 10,
    min_resolution: Tuple[int, int] = (160, 160),
    brightness_range: Tuple[float, float] = (20.0, 235.0),
    min_blur_score: float = 20.0,
) -> Dict:
    """
    Lightweight sanity checks on the input video.

    Returns a dict of QualityReport.

    Notes:
    - These are heuristics; they should produce warnings rather than hard failures
      unless the video is unreadable or has too few frames.
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict = {}

    cap = None
    try:
        cap = open_video(video_path)
        meta = read_video_meta(video_path, cap=cap)

        stats["fps"] = meta.fps
        stats["frame_count"] = meta.frame_count
        stats["width"] = meta.width
        stats["height"] = meta.height
        stats["duration_seconds"] = meta.duration_seconds

        if meta.frame_count < min_frame_count:
            errors.append(f"Too few frames: {meta.frame_count} < {min_frame_count}")

        if meta.width < min_resolution[0] or meta.height < min_resolution[1]:
            warnings.append(f"Low resolution: {meta.width}x{meta.height} < {min_resolution[0]}x{min_resolution[1]}")

        # Sample frames uniformly for quick checks
        if meta.frame_count > 0 and sample_frames > 0:
            idxs = []
            if sample_frames >= meta.frame_count:
                idxs = list(range(meta.frame_count))
            else:
                for i in range(sample_frames):
                    pos = round(i * (meta.frame_count - 1) / (sample_frames - 1))
                    idxs.append(int(pos))

            brightness_vals: List[float] = []
            blur_vals: List[float] = []
            unreadable = 0

            for idx in idxs:
                ok, frame = read_frame_at(cap, idx)
                if not ok or frame is None:
                    unreadable += 1
                    continue
                b = _mean_brightness(frame)
                s = _blur_score_laplacian(frame)
                brightness_vals.append(b)
                blur_vals.append(s)

            stats["sampled_frames"] = idxs
            stats["unreadable_samples"] = unreadable
            if brightness_vals:
                stats["brightness_mean"] = float(np.mean(brightness_vals))
                stats["brightness_min"] = float(np.min(brightness_vals))
                stats["brightness_max"] = float(np.max(brightness_vals))
            if blur_vals:
                stats["blur_mean"] = float(np.mean(blur_vals))
                stats["blur_min"] = float(np.min(blur_vals))
                stats["blur_max"] = float(np.max(blur_vals))

            # Brightness warning
            if brightness_vals:
                bmean = float(np.mean(brightness_vals))
                if bmean < brightness_range[0]:
                    warnings.append(f"Video appears very dark (mean brightness {bmean:.1f})")
                if bmean > brightness_range[1]:
                    warnings.append(f"Video appears very bright (mean brightness {bmean:.1f})")

            # Blur warning
            if blur_vals:
                smean = float(np.mean(blur_vals))
                if smean < min_blur_score:
                    warnings.append(f"Video may be blurry (mean blur score {smean:.1f} < {min_blur_score})")

        ok = len(errors) == 0
        report = QualityReport(ok=ok, errors=errors, warnings=warnings, stats=stats)
        return asdict(report)

    finally:
        if cap is not None:
            cap.release()
