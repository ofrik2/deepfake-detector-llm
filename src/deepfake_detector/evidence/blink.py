from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import cv2
import numpy as np


@dataclass(frozen=True)
class BlinkEvidence:
    blink_detected: bool
    estimated_blink_count: int
    blink_confidence: float
    eye_openness_series: List[float]
    openness_threshold: float
    method: str


def _eye_openness_proxy(gray_roi: np.ndarray) -> float:
    """
    Openness proxy: Laplacian variance (edge energy).
    Cast to uint8 for OpenCV compatibility.
    """
    if gray_roi.size == 0:
        return 0.0

    # Ensure OpenCV-friendly dtype
    if gray_roi.dtype != np.uint8:
        roi = np.clip(gray_roi, 0, 255).astype(np.uint8)
    else:
        roi = gray_roi

    lap = cv2.Laplacian(roi, cv2.CV_64F)
    return float(lap.var())



def _smooth_1d(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return x.copy()
    y = x.copy()
    for i in range(1, len(x) - 1):
        y[i] = np.median([x[i - 1], x[i], x[i + 1]])
    return y


def _count_dip_events(series: np.ndarray, thresh: float) -> int:
    n = len(series)
    if n < 3:
        return 0
    count = 0
    for i in range(1, n - 1):
        if series[i] < thresh and series[i] <= series[i - 1] and series[i] <= series[i + 1]:
            if series[i - 1] > thresh or series[i + 1] > thresh:
                count += 1
    return count


def compute_blink_evidence_from_eyes_roi_series(eyes_rois_gray: List[np.ndarray]) -> Dict:
    """
    Main API expected by basic_signals.py

    Input:
      eyes_rois_gray: list of grayscale eye-ROI images per sampled frame

    Output:
      dict (JSON-serializable) with blink fields
    """
    if not eyes_rois_gray:
        return asdict(
            BlinkEvidence(
                blink_detected=False,
                estimated_blink_count=0,
                blink_confidence=0.0,
                eye_openness_series=[],
                openness_threshold=0.0,
                method="laplacian_var",
            )
        )

    openness = np.array([_eye_openness_proxy(roi) for roi in eyes_rois_gray], dtype=np.float64)
    openness_s = _smooth_1d(openness)

    rng = float(openness_s.max() - openness_s.min())
    if rng < 1e-6:
        mu = float(openness_s.mean())
        return asdict(
            BlinkEvidence(
                blink_detected=False,
                estimated_blink_count=0,
                blink_confidence=0.0,
                eye_openness_series=[float(v) for v in openness_s],
                openness_threshold=mu,
                method="laplacian_var",
            )
        )

    mu = float(openness_s.mean())
    sd = float(openness_s.std(ddof=0))

    thresh = mu - 0.8 * sd
    if sd < 1e-6:
        thresh = float(np.percentile(openness_s, 20))

    blink_count = _count_dip_events(openness_s, thresh)

    dip_strength = float((mu - openness_s.min()) / (sd + 1e-9))
    conf = 1.0 - np.exp(-0.6 * max(0.0, dip_strength))
    if blink_count == 0:
        conf *= 0.6

    return asdict(
        BlinkEvidence(
            blink_detected=blink_count > 0,
            estimated_blink_count=int(blink_count),
            blink_confidence=float(max(0.0, min(1.0, conf))),
            eye_openness_series=[float(v) for v in openness_s],
            openness_threshold=float(thresh),
            method="laplacian_var",
        )
    )
