"""
basic_signals.py

Compute simple, explainable "evidence signals" from extracted video frames.

Signals:
1) Global motion proxy: mean absolute difference between consecutive sampled frames.
2) Face-region motion proxies:
   - Mouth ROI motion (lower-face region)
   - Eyes ROI motion (upper-face region)
3) Blink evidence (heuristic) computed from per-frame eyes ROI series.
4) Face bbox jitter stats (stability of face detection/tracking)
5) Face-boundary evidence (forensic-style):
   - compares edge-energy near face boundary vs interior face region
6) Boundary calibration vs background:
   - compute the same boundary ratio on a background patch
   - report face_over_bg and face_minus_bg

This module produces evidence only; it does NOT classify real vs manipulated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .blink import compute_blink_evidence_from_eyes_roi_series


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
    roi_method: str  # "haar_face_per_frame" or "fallback_center"
    face_bbox: Optional[Dict[str, int]]  # representative bbox if available
    face_bbox_series: List[Optional[Dict[str, int]]]  # per-frame bbox dict or None
    mouth_roi: Dict[str, int]  # representative ROI in original coords
    eyes_roi: Dict[str, int]   # representative ROI in original coords

    # Face bbox jitter (stability)
    face_bbox_none_ratio: float
    face_bbox_center_jitter_mean: float
    face_bbox_size_jitter_mean: float

    # Global motion
    global_motion_mean: float
    global_motion_min: float
    global_motion_max: float
    global_motion_p95: float
    global_per_pair_motion: List[float]

    # Mouth motion
    mouth_motion_mean: float
    mouth_motion_min: float
    mouth_motion_max: float
    mouth_motion_p95: float
    mouth_motion_max_over_mean: float
    mouth_per_pair_motion: List[float]

    # Eyes motion
    eyes_motion_mean: float
    eyes_motion_min: float
    eyes_motion_max: float
    eyes_motion_p95: float
    eyes_motion_max_over_mean: float
    eyes_per_pair_motion: List[float]
    blink_like_events: int

    # Blink evidence (heuristic)
    blink_detected: bool
    estimated_blink_count: int
    blink_confidence: float
    eye_openness_series: List[float]
    openness_threshold: float
    blink_method: str
    eye_openness_range: float
    blink_dip_fraction: float

    # Boundary evidence (face)
    boundary_face_ratio_mean: float
    boundary_face_ratio_std: float
    boundary_face_ratio_series: List[float]
    boundary_method: str

    # Boundary calibration vs background
    boundary_bg_ratio_mean: float
    boundary_bg_ratio_std: float
    boundary_face_over_bg_mean: float
    boundary_face_minus_bg_mean: float

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


def _mean_abs_diff_allow_mismatch(a: np.ndarray, b: np.ndarray) -> float:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    w = min(wa, wb)
    if h <= 0 or w <= 0:
        return 0.0
    a2 = a[:h, :w]
    b2 = b[:h, :w]
    return float(np.mean(np.abs(a2 - b2)))


def _crop_roi(gray: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    H, W = gray.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return gray[y:y + h, x:x + w]


# -------------------------
# ROI detection (Haar + fallback)
# -------------------------

def _detect_face_bbox_haar(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(90, 90),
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
    H, W = frame_bgr.shape[:2]
    eyes = {
        "x": int(0.20 * W),
        "y": int(0.18 * H),
        "w": int(0.60 * W),
        "h": int(0.22 * H),
    }
    mouth = {
        "x": int(0.25 * W),
        "y": int(0.58 * H),
        "w": int(0.50 * W),
        "h": int(0.22 * H),
    }
    return mouth, eyes


# -------------------------
# Jitter evidence
# -------------------------

def _bbox_jitter_stats(face_bbox_series: List[Optional[Dict[str, int]]]) -> Dict[str, float]:
    bbs = [bb for bb in face_bbox_series if bb is not None]
    total = len(face_bbox_series)
    valid = len(bbs)

    if total == 0:
        return {
            "face_bbox_none_ratio": 0.0,
            "face_bbox_center_jitter_mean": 0.0,
            "face_bbox_size_jitter_mean": 0.0,
        }

    if valid < 2:
        return {
            "face_bbox_none_ratio": float((total - valid) / float(total)),
            "face_bbox_center_jitter_mean": 0.0,
            "face_bbox_size_jitter_mean": 0.0,
        }

    none_ratio = (total - valid) / float(total)

    centers = []
    sizes = []
    for bb in bbs:
        cx = bb["x"] + bb["w"] / 2.0
        cy = bb["y"] + bb["h"] / 2.0
        centers.append((cx, cy))
        sizes.append((bb["w"], bb["h"]))

    center_d = []
    size_d = []
    for i in range(len(centers) - 1):
        (cx1, cy1), (cx2, cy2) = centers[i], centers[i + 1]
        (w1, h1), (w2, h2) = sizes[i], sizes[i + 1]

        norm = max(1e-6, (w1 + h1) / 2.0)
        center_d.append((abs(cx2 - cx1) + abs(cy2 - cy1)) / norm)

        size_norm = max(1e-6, (w1 + h1) / 2.0)
        size_d.append((abs(w2 - w1) + abs(h2 - h1)) / size_norm)

    return {
        "face_bbox_none_ratio": float(none_ratio),
        "face_bbox_center_jitter_mean": float(np.mean(center_d)) if center_d else 0.0,
        "face_bbox_size_jitter_mean": float(np.mean(size_d)) if size_d else 0.0,
    }


# -------------------------
# Boundary evidence
# -------------------------

def _laplacian_var_u8(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    if gray.dtype != np.uint8:
        g = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        g = gray
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap.var())


def _boundary_edge_ratio_for_roi(
    gray: np.ndarray,
    roi: Dict[str, int],
    shrink_frac: float = 0.12,
) -> float:
    """
    Compute edge-energy ratio: ring (near ROI boundary) / inner.
    """
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    if w < 10 or h < 10:
        return 0.0

    dx = int(round(shrink_frac * w))
    dy = int(round(shrink_frac * h))
    inner = {
        "x": x + dx,
        "y": y + dy,
        "w": max(1, w - 2 * dx),
        "h": max(1, h - 2 * dy),
    }

    outer_img = _crop_roi(gray, roi)
    inner_img = _crop_roi(gray, inner)

    ring = outer_img.copy()

    ix = inner["x"] - roi["x"]
    iy = inner["y"] - roi["y"]
    iw = inner["w"]
    ih = inner["h"]

    ix = max(0, min(ix, ring.shape[1] - 1))
    iy = max(0, min(iy, ring.shape[0] - 1))
    iw = max(1, min(iw, ring.shape[1] - ix))
    ih = max(1, min(ih, ring.shape[0] - iy))

    ring[iy:iy + ih, ix:ix + iw] = 0

    ring_energy = _laplacian_var_u8(ring)
    inner_energy = _laplacian_var_u8(inner_img)
    return float(ring_energy / (inner_energy + 1e-9))


def _background_roi(proc_w: int, proc_h: int) -> Dict[str, int]:
    """
    Fixed background patch (processed coords), top-left-ish.
    Avoid center to reduce overlap with face.
    """
    x = int(0.05 * proc_w)
    y = int(0.05 * proc_h)
    w = int(0.30 * proc_w)
    h = int(0.30 * proc_h)
    w = max(10, w)
    h = max(10, h)
    if x + w >= proc_w:
        x = max(0, proc_w - w - 1)
    if y + h >= proc_h:
        y = max(0, proc_h - h - 1)
    return {"x": x, "y": y, "w": w, "h": h}


# -------------------------
# Main API
# -------------------------

def compute_basic_signals(
    manifest_path: str,
    *,
    frames_dir: Optional[str] = None,
    diff_resize_max: int = 256,
) -> Dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    frames_dir_resolved = _resolve_frames_dir(manifest_path, frames_dir)

    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError("Manifest contains no frames.")

    filenames = [f["filename"] for f in frames]
    notes: List[str] = []

    bgr_frames: List[np.ndarray] = [_read_frame_bgr(frames_dir_resolved, fn) for fn in filenames]

    # Per-frame face bbox detection with carry-forward fallback
    face_bbox_series_raw: List[Optional[Tuple[int, int, int, int]]] = []
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    for bgr in bgr_frames:
        bb = _detect_face_bbox_haar(bgr)
        if bb is None:
            bb = last_bbox
        else:
            last_bbox = bb
        face_bbox_series_raw.append(bb)

    any_face = any(bb is not None for bb in face_bbox_series_raw)
    if any_face:
        roi_method = "haar_face_per_frame"
    else:
        roi_method = "fallback_center"
        notes.append("Face detection failed on all frames; using center-based fallback ROIs.")

    gray_frames: List[np.ndarray] = [_preprocess_gray(bgr, resize_max=diff_resize_max) for bgr in bgr_frames]

    # Scaling between original and processed gray
    orig_h, orig_w = bgr_frames[0].shape[:2]
    proc_h, proc_w = gray_frames[0].shape[:2]
    sx = proc_w / float(orig_w)
    sy = proc_h / float(orig_h)

    def _scale_roi(roi: Dict[str, int]) -> Dict[str, int]:
        return {
            "x": int(round(roi["x"] * sx)),
            "y": int(round(roi["y"] * sy)),
            "w": max(1, int(round(roi["w"] * sx))),
            "h": max(1, int(round(roi["h"] * sy))),
        }

    # Per-frame ROI series + bbox dict series (original coords)
    mouth_roi_series_s: List[Dict[str, int]] = []
    eyes_roi_series_s: List[Dict[str, int]] = []
    face_roi_series_s: List[Dict[str, int]] = []  # scaled face bbox for boundary
    face_bbox_series_dict: List[Optional[Dict[str, int]]] = []

    if not any_face:
        fallback_mouth_o, fallback_eyes_o = _fallback_rois_center(bgr_frames[0])
        fallback_face_o = {"x": int(0.15 * orig_w), "y": int(0.10 * orig_h), "w": int(0.70 * orig_w), "h": int(0.80 * orig_h)}
        for _ in bgr_frames:
            mouth_roi_series_s.append(_scale_roi(fallback_mouth_o))
            eyes_roi_series_s.append(_scale_roi(fallback_eyes_o))
            face_roi_series_s.append(_scale_roi(fallback_face_o))
            face_bbox_series_dict.append(None)
    else:
        for bb in face_bbox_series_raw:
            if bb is None:
                fallback_mouth_o, fallback_eyes_o = _fallback_rois_center(bgr_frames[0])
                fallback_face_o = {"x": int(0.15 * orig_w), "y": int(0.10 * orig_h), "w": int(0.70 * orig_w), "h": int(0.80 * orig_h)}
                mouth_roi_series_s.append(_scale_roi(fallback_mouth_o))
                eyes_roi_series_s.append(_scale_roi(fallback_eyes_o))
                face_roi_series_s.append(_scale_roi(fallback_face_o))
                face_bbox_series_dict.append(None)
            else:
                mouth_o = _mouth_roi_from_face(bb)
                eyes_o = _eyes_roi_from_face(bb)
                face_o = {"x": bb[0], "y": bb[1], "w": bb[2], "h": bb[3]}
                mouth_roi_series_s.append(_scale_roi(mouth_o))
                eyes_roi_series_s.append(_scale_roi(eyes_o))
                face_roi_series_s.append(_scale_roi(face_o))
                face_bbox_series_dict.append(face_o)

    # Representative bbox/ROIs
    rep_idx = next((i for i, bb in enumerate(face_bbox_series_raw) if bb is not None), None)
    if rep_idx is not None:
        rep_bb = face_bbox_series_raw[rep_idx]
        rep_face_bbox_dict = {"x": rep_bb[0], "y": rep_bb[1], "w": rep_bb[2], "h": rep_bb[3]}
        rep_mouth_roi = _mouth_roi_from_face(rep_bb)
        rep_eyes_roi = _eyes_roi_from_face(rep_bb)
    else:
        rep_face_bbox_dict = None
        rep_mouth_roi, rep_eyes_roi = _fallback_rois_center(bgr_frames[0])

    # Jitter
    jitter = _bbox_jitter_stats(face_bbox_series_dict)
    notes.append(f"Face bbox none ratio: {jitter['face_bbox_none_ratio']:.2f} (fraction of frames with no detected face).")
    notes.append(f"Face bbox center jitter mean (normalized): {jitter['face_bbox_center_jitter_mean']:.3f}.")
    notes.append(f"Face bbox size jitter mean (normalized): {jitter['face_bbox_size_jitter_mean']:.3f}.")

    # Motion diffs
    global_per: List[float] = []
    mouth_per: List[float] = []
    eyes_per: List[float] = []

    for i in range(len(gray_frames) - 1):
        a = gray_frames[i]
        b = gray_frames[i + 1]

        global_per.append(_mean_abs_diff(a, b))

        a_m = _crop_roi(a, mouth_roi_series_s[i])
        b_m = _crop_roi(b, mouth_roi_series_s[i + 1])
        mouth_per.append(_mean_abs_diff_allow_mismatch(a_m, b_m))

        a_e = _crop_roi(a, eyes_roi_series_s[i])
        b_e = _crop_roi(b, eyes_roi_series_s[i + 1])
        eyes_per.append(_mean_abs_diff_allow_mismatch(a_e, b_e))

    def summarize(vals: List[float]) -> Tuple[float, float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.min()), float(arr.max()), float(np.percentile(arr, 95))

    g_mean, g_min, g_max, g_p95 = summarize(global_per)
    m_mean, m_min, m_max, m_p95 = summarize(mouth_per)
    e_mean, e_min, e_max, e_p95 = summarize(eyes_per)

    mouth_max_over_mean = float(m_max / (m_mean + 1e-9)) if m_mean > 0 else 0.0
    eyes_max_over_mean = float(e_max / (e_mean + 1e-9)) if e_mean > 0 else 0.0

    if e_mean > 0:
        ratio = m_mean / e_mean
        notes.append(f"Mouth-vs-eyes motion ratio: {ratio:.2f} (higher means mouth changes more than eyes).")
    else:
        notes.append("Eye motion mean is ~0; eye region appears extremely static in sampled frames.")

    notes.append(f"Eyes motion p95: {e_p95:.3f}; max/mean: {eyes_max_over_mean:.2f}.")
    notes.append(f"Mouth motion p95: {m_p95:.3f}; max/mean: {mouth_max_over_mean:.2f}.")

    # Blink evidence
    def _standardize_eye_roi(roi: np.ndarray, size=(64, 32)) -> np.ndarray:
        # ensure float32 and fixed size so openness values are comparable across frames
        if roi.size == 0:
            return np.zeros((size[1], size[0]), dtype=np.float32)
        roi_f = roi.astype(np.float32)
        return cv2.resize(roi_f, size, interpolation=cv2.INTER_AREA)

    eyes_rois_gray = []
    for i in range(len(gray_frames)):
        roi = _crop_roi(gray_frames[i], eyes_roi_series_s[i])
        roi = _standardize_eye_roi(roi, size=(64, 32))
        eyes_rois_gray.append(roi)

    blink = compute_blink_evidence_from_eyes_roi_series(eyes_rois_gray)

    if blink.get("blink_detected") is False:
        notes.append("No blink dip detected in sampled frames (heuristic).")
    op_series = list(blink.get("eye_openness_series", []))
    op = np.asarray(op_series, dtype=np.float64) if op_series else np.asarray([], dtype=np.float64)

    # Normalize openness series so range is comparable (robust z-score)
    med = float(np.median(op))
    mad = float(np.median(np.abs(op - med))) + 1e-9
    op_n = (op - med) / mad

    if op.size >= 2:
        eye_openness_range = float(op_n.max() - op_n.min())
        mean = float(op.mean())
        std = float(op.std())

        if std > 1e-9:
            dip_thr = float(op_n.mean() - 1.5 * op_n.std())
            is_dip = op_n < dip_thr

            # count blink-like events: >=2 consecutive dip frames
            blink_like_events = 0
            i = 0
            while i < len(is_dip):
                if is_dip[i]:
                    j = i
                    while j < len(is_dip) and is_dip[j]:
                        j += 1
                    if (j - i) >= 2:
                        blink_like_events += 1
                    i = j
                else:
                    i += 1

            blink_dip_fraction = float(np.mean(is_dip))
        else:
            blink_like_events = 0
            blink_dip_fraction = 0.0
    else:
        eye_openness_range = 0.0
        blink_like_events = 0
        blink_dip_fraction = 0.0


    notes.append(f"Eye openness range (max-min): {eye_openness_range:.3f}.")
    notes.append(f"Blink dip fraction: {blink_dip_fraction:.3f} (fraction of frames with strong openness drop).")

    # Boundary evidence (face) + background calibration
    face_boundary_series: List[float] = []
    bg_boundary_series: List[float] = []

    bg_roi = _background_roi(proc_w, proc_h)

    for i in range(len(gray_frames)):
        face_boundary_series.append(_boundary_edge_ratio_for_roi(gray_frames[i], face_roi_series_s[i], shrink_frac=0.12))
        bg_boundary_series.append(_boundary_edge_ratio_for_roi(gray_frames[i], bg_roi, shrink_frac=0.12))

    face_mean = float(np.mean(face_boundary_series)) if face_boundary_series else 0.0
    face_std = float(np.std(face_boundary_series)) if face_boundary_series else 0.0
    bg_mean = float(np.mean(bg_boundary_series)) if bg_boundary_series else 0.0
    bg_std = float(np.std(bg_boundary_series)) if bg_boundary_series else 0.0

    face_over_bg = float(face_mean / (bg_mean + 1e-9)) if bg_mean > 0 else 0.0
    face_minus_bg = float(face_mean - bg_mean)

    notes.append(f"Face-boundary ratio mean: {face_mean:.3f}, std: {face_std:.3f}.")
    notes.append(f"Background-boundary ratio mean: {bg_mean:.3f}, std: {bg_std:.3f}.")
    notes.append(f"Boundary face_over_bg mean: {face_over_bg:.3f}; face_minus_bg mean: {face_minus_bg:.3f}.")

    result = EvidenceResult(
        manifest_path=str(Path(manifest_path)),
        frames_dir=frames_dir_resolved,
        sampled_frame_count=len(filenames),
        sampled_frame_files=filenames,

        roi_method=roi_method,
        face_bbox=rep_face_bbox_dict,
        face_bbox_series=face_bbox_series_dict,
        mouth_roi=rep_mouth_roi,
        eyes_roi=rep_eyes_roi,

        face_bbox_none_ratio=float(jitter["face_bbox_none_ratio"]),
        face_bbox_center_jitter_mean=float(jitter["face_bbox_center_jitter_mean"]),
        face_bbox_size_jitter_mean=float(jitter["face_bbox_size_jitter_mean"]),

        global_motion_mean=g_mean,
        global_motion_min=g_min,
        global_motion_max=g_max,
        global_motion_p95=g_p95,
        global_per_pair_motion=global_per,

        mouth_motion_mean=m_mean,
        mouth_motion_min=m_min,
        mouth_motion_max=m_max,
        mouth_motion_p95=m_p95,
        mouth_motion_max_over_mean=mouth_max_over_mean,
        mouth_per_pair_motion=mouth_per,

        eyes_motion_mean=e_mean,
        eyes_motion_min=e_min,
        eyes_motion_max=e_max,
        eyes_motion_p95=e_p95,
        eyes_motion_max_over_mean=eyes_max_over_mean,
        eyes_per_pair_motion=eyes_per,
        eye_openness_range=eye_openness_range,
        blink_dip_fraction=blink_dip_fraction,
        blink_like_events=blink_like_events,

        blink_detected=bool(blink.get("blink_detected", False)),
        estimated_blink_count=int(blink.get("estimated_blink_count", 0)),
        blink_confidence=float(blink.get("blink_confidence", 0.0)),
        eye_openness_series=list(blink.get("eye_openness_series", [])),
        openness_threshold=float(blink.get("openness_threshold", 0.0)),
        blink_method=str(blink.get("method", "unknown")),

        boundary_face_ratio_mean=face_mean,
        boundary_face_ratio_std=face_std,
        boundary_face_ratio_series=face_boundary_series,
        boundary_method="laplacian_var_ring_over_inner",

        boundary_bg_ratio_mean=bg_mean,
        boundary_bg_ratio_std=bg_std,
        boundary_face_over_bg_mean=face_over_bg,
        boundary_face_minus_bg_mean=face_minus_bg,

        notes=notes,
    )

    return asdict(result)


def save_basic_signals(evidence: Dict[str, Any], out_path: str) -> None:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
