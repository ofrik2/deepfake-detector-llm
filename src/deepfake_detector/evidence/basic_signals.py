"""
basic_signals.py

Compute simple, explainable "evidence signals" from extracted video frames.

This module sits between:
- deepfake_detector.video.*  (video -> frames + manifest)
and
- deepfake_detector.llm.*    (reasoning / prompting)

Design goals:
- Deterministic: same inputs -> same outputs
- Lightweight: OpenCV + NumPy only
- Explainable: signals are easy to describe in the report
- Modular: easy to add more signals without changing the contract

Current signals:
1) Global motion proxy: mean absolute difference between consecutive sampled frames.
2) Face-region motion proxies:
   - Mouth ROI motion (lower-face region)
   - Eyes ROI motion (upper-face region)

ROI detection:
- Primary: Haar cascade frontal face detection (OpenCV built-in).
- Fallback: center-based ROIs if face detection fails.

Important:
This module does NOT decide "deepfake vs real". It only computes evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class EvidenceResult:
    manifest_path: str
    frames_dir: str
    sampled_frame_count: int
    sampled_frame_files: List[str]

    # ROI info (for explainability/debugging)
    roi_method: str  # "haar_face" or "fallback_center"
    face_bbox: Optional[Dict[str, int]]  # x,y,w,h if available
    mouth_roi: Dict[str, int]  # x,y,w,h in frame coordinates
    eyes_roi: Dict[str, int]   # x,y,w,h in frame coordinates

    # Global motion
    global_motion_mean: float
    global_motion_min: float
    global_motion_max: float
    global_per_pair_motion: List[float]

    # Mouth motion
    mouth_motion_mean: float
    mouth_motion_min: float
    mouth_motion_max: float
    mouth_per_pair_motion: List[float]

    # Eyes motion
    eyes_motion_mean: float
    eyes_motion_min: float
    eyes_motion_max: float
    eyes_per_pair_motion: List[float]

    notes: List[str]


# -------------------------
# Manifest + IO
# -------------------------

def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"frames_manifest.json not found: {manifest_path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_frames_dir(manifest_path: str, frames_dir: Optional[str]) -> str:
    if frames_dir:
        return str(Path(frames_dir))
    return str(Path(manifest_path).resolve().parent)


def _read_frame_bgr(frames_dir: str, filename: str) -> np.ndarray:
    path = Path(frames_dir) / filename
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read frame image: {path}")
    return img


# -------------------------
# Preprocessing + diffs
# -------------------------

def _preprocess_gray(frame_bgr: np.ndarray, resize_max: int = 256) -> np.ndarray:
    """
    Convert to grayscale float32 and optionally downscale.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if resize_max > 0:
        h, w = gray.shape[:2]
        m = max(h, w)
        if m > resize_max:
            scale = resize_max / float(m)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return gray.astype(np.float32)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Frame shapes do not match: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)))


def _crop_roi(gray: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    H, W = gray.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return gray[y:y + h, x:x + w]


# -------------------------
# ROI detection (A + fallback)
# -------------------------

def _detect_face_bbox_haar(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (x,y,w,h) for the largest detected frontal face, or None if not found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def _mouth_roi_from_face(face_bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
    x, y, w, h = face_bbox
    mx = x + int(0.20 * w)
    my = y + int(0.62 * h)
    mw = int(0.60 * w)
    mh = int(0.22 * h)
    return {"x": mx, "y": my, "w": mw, "h": mh}


def _eyes_roi_from_face(face_bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
    x, y, w, h = face_bbox
    ex = x + int(0.15 * w)
    ey = y + int(0.22 * h)
    ew = int(0.70 * w)
    eh = int(0.22 * h)
    return {"x": ex, "y": ey, "w": ew, "h": eh}


def _fallback_rois_center(frame_bgr: np.ndarray) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Center-based fallback ROIs (works even if face detection fails).
    Assumes a typical portrait/centered talking-head.
    """
    H, W = frame_bgr.shape[:2]

    # Eyes region: upper-middle
    eyes = {
        "x": int(0.20 * W),
        "y": int(0.18 * H),
        "w": int(0.60 * W),
        "h": int(0.22 * H),
    }

    # Mouth region: lower-middle
    mouth = {
        "x": int(0.25 * W),
        "y": int(0.58 * H),
        "w": int(0.50 * W),
        "h": int(0.22 * H),
    }
    return mouth, eyes


# -------------------------
# Main API
# -------------------------

def compute_basic_signals(
    manifest_path: str,
    *,
    frames_dir: Optional[str] = None,
    diff_resize_max: int = 256,
) -> Dict[str, Any]:
    """
    Compute basic evidence signals from extracted frames.

    Parameters
    ----------
    manifest_path:
        Path to frames_manifest.json created by extract_frames().
    frames_dir:
        Directory containing frame JPG files. If omitted, uses manifest directory.
    diff_resize_max:
        Downscale frames for diff computation only.

    Returns
    -------
    dict (JSON-serializable)
    """
    manifest = _load_manifest(manifest_path)
    frames_dir_resolved = _resolve_frames_dir(manifest_path, frames_dir)

    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError("Manifest contains no frames.")

    filenames = [f["filename"] for f in frames]
    notes: List[str] = []

    if len(filenames) < 2:
        notes.append("Only one frame in manifest; motion scores require >= 2 frames.")

    # Load all frames (BGR)
    bgr_frames: List[np.ndarray] = [_read_frame_bgr(frames_dir_resolved, fn) for fn in filenames]

    # Detect face bbox on the FIRST sampled frame (stable across the sample set)
    face_bbox = _detect_face_bbox_haar(bgr_frames[0])

    if face_bbox is not None:
        roi_method = "haar_face"
        mouth_roi = _mouth_roi_from_face(face_bbox)
        eyes_roi = _eyes_roi_from_face(face_bbox)
        face_bbox_dict = {"x": face_bbox[0], "y": face_bbox[1], "w": face_bbox[2], "h": face_bbox[3]}
    else:
        roi_method = "fallback_center"
        mouth_roi, eyes_roi = _fallback_rois_center(bgr_frames[0])
        face_bbox_dict = None
        notes.append("Face detection failed; using center-based fallback ROIs.")

    # Preprocess to grayscale (optionally downscaled)
    gray_frames: List[np.ndarray] = [_preprocess_gray(bgr, resize_max=diff_resize_max) for bgr in bgr_frames]

    # IMPORTANT:
    # If we downscale for diff, we must also scale ROIs accordingly.
    # Compute scale factors between original frame and processed gray frame.
    orig_h, orig_w = bgr_frames[0].shape[:2]
    proc_h, proc_w = gray_frames[0].shape[:2]
    sx = proc_w / float(orig_w)
    sy = proc_h / float(orig_h)

    def scale_roi(roi: Dict[str, int]) -> Dict[str, int]:
        return {
            "x": int(round(roi["x"] * sx)),
            "y": int(round(roi["y"] * sy)),
            "w": max(1, int(round(roi["w"] * sx))),
            "h": max(1, int(round(roi["h"] * sy))),
        }

    mouth_roi_s = scale_roi(mouth_roi)
    eyes_roi_s = scale_roi(eyes_roi)

    # Compute per-pair motion scores
    global_per: List[float] = []
    mouth_per: List[float] = []
    eyes_per: List[float] = []

    for i in range(len(gray_frames) - 1):
        a = gray_frames[i]
        b = gray_frames[i + 1]

        global_per.append(_mean_abs_diff(a, b))

        a_m = _crop_roi(a, mouth_roi_s)
        b_m = _crop_roi(b, mouth_roi_s)
        mouth_per.append(_mean_abs_diff(a_m, b_m))

        a_e = _crop_roi(a, eyes_roi_s)
        b_e = _crop_roi(b, eyes_roi_s)
        eyes_per.append(_mean_abs_diff(a_e, b_e))

    def summarize(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))

    g_mean, g_min, g_max = summarize(global_per)
    m_mean, m_min, m_max = summarize(mouth_per)
    e_mean, e_min, e_max = summarize(eyes_per)

    # Add lightweight interpretive notes (not a decision)
    if global_per:
        if g_mean < 2.0:
            notes.append("Low overall motion across sampled frames (global mean abs diff < 2.0).")
        elif g_mean > 15.0:
            notes.append("High overall motion across sampled frames (global mean abs diff > 15.0).")

    if mouth_per and eyes_per:
        # Ratio to highlight "mouth moves, eyes don't" pattern
        if e_mean > 0:
            ratio = m_mean / e_mean
            notes.append(f"Mouth-vs-eyes motion ratio: {ratio:.2f} (higher means mouth changes more than eyes).")
            if ratio >= 2.5:
                notes.append("Mouth region changes substantially more than eye region (consistent with speaking).")
            if e_mean < 1.5:
                notes.append("Eye-region motion is very low across sampled frames (possible low blinking or static eyes).")
        else:
            notes.append("Eye motion mean is ~0; eye region appears extremely static in sampled frames.")

    result = EvidenceResult(
        manifest_path=str(Path(manifest_path)),
        frames_dir=frames_dir_resolved,
        sampled_frame_count=len(filenames),
        sampled_frame_files=filenames,
        roi_method=roi_method,
        face_bbox=face_bbox_dict,
        mouth_roi=mouth_roi,
        eyes_roi=eyes_roi,
        global_motion_mean=g_mean,
        global_motion_min=g_min,
        global_motion_max=g_max,
        global_per_pair_motion=global_per,
        mouth_motion_mean=m_mean,
        mouth_motion_min=m_min,
        mouth_motion_max=m_max,
        mouth_per_pair_motion=mouth_per,
        eyes_motion_mean=e_mean,
        eyes_motion_min=e_min,
        eyes_motion_max=e_max,
        eyes_per_pair_motion=eyes_per,
        notes=notes,
    )

    return asdict(result)


def save_basic_signals(evidence: Dict[str, Any], out_path: str) -> None:
    """
    Save evidence dict to JSON.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
