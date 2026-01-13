"""
Synthetic video generator (v0–v4) using a static real face image.

SEE docs/VIDEO_GENERATION.md

This generator creates deterministic, photo-based synthetic videos by loading a
static face image and applying controlled, explainable temporal artifacts.

All variants require --base-image (a frontal face photo).
Outputs:
- MP4 video
- TXT ground-truth description (for documentation/evaluation only)

# Put a face photo here first:
# assets/base_faces/face_01.jpg

python src/deepfake_detector/generation/opencv_generator.py --variant v0 \
  --base-image assets/base_faces/face_01.jpg \
  --out-video assets/videos/v0.mp4 \
  --out-desc assets/videos/v0.txt

python src/deepfake_detector/generation/opencv_generator.py --variant v1 \
  --base-image assets/base_faces/face_01.jpg \
  --out-video assets/videos/v1.mp4 \
  --out-desc assets/videos/v1.txt

python src/deepfake_detector/generation/opencv_generator.py --variant v2 \
  --base-image assets/base_faces/face_01.jpg \
  --out-video assets/videos/v2.mp4 \
  --out-desc assets/videos/v2.txt

python src/deepfake_detector/generation/opencv_generator.py --variant v3 \
  --base-image assets/base_faces/face_01.jpg \
  --out-video assets/videos/v3.mp4 \
  --out-desc assets/videos/v3.txt

python src/deepfake_detector/generation/opencv_generator.py --variant v4 \
  --base-image assets/base_faces/face_01.jpg \
  --out-video assets/videos/v4.mp4 \
  --out-desc assets/videos/v4.txt

"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


# =========================
# Variants
# =========================

class VideoVariant(str, Enum):
    """
    v0–v4 photo-based variants (single static face image as the base).

    v0 (FAKE): Mouth moves smoothly; eyes never blink (perfectly static).
    v1 (FAKE): Mouth moves smoothly; blinking is perfectly periodic (unnatural regularity).
    v2 (FAKE): Mouth motion is jittery/discontinuous (frame-to-frame jumps).
    v3 (FAKE): Lighting flickers periodically (scene brightness inconsistency).
    v4 (REAL-ish baseline): Mouth moves smoothly; blinking is pseudo-random; tiny smooth head drift.
    """
    V0 = "v0"
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"


@dataclass(frozen=True)
class VideoGenSpec:
    width: int = 512
    height: int = 512
    fps: int = 30
    duration_seconds: int = 10
    seed: int = 123


# =========================
# Utilities
# =========================

def _validate_spec(spec: VideoGenSpec) -> None:
    if spec.width <= 0 or spec.height <= 0:
        raise ValueError(f"Invalid frame size: {spec.width}x{spec.height}")
    if spec.fps <= 0:
        raise ValueError(f"Invalid fps: {spec.fps}")
    if spec.duration_seconds <= 0:
        raise ValueError(f"Invalid duration_seconds: {spec.duration_seconds}")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _clamp_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _load_base_image(path: str, width: int, height: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read base image: {path}")
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _detect_face_bbox_bgr(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detect the largest frontal face using Haar cascades (OpenCV built-in).
    Requires a reasonably frontal, clear face image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120),
    )
    if len(faces) == 0:
        raise RuntimeError(
            "No face detected in base image. "
            "Use a clearer frontal face photo, or add manual ROI support."
        )
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])  # largest
    return int(x), int(y), int(w), int(h)


def _mouth_roi_from_face(face_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Deterministic mouth ROI as a fraction of the detected face box.
    Tuned for typical frontal faces.
    """
    x, y, w, h = face_bbox
    mx = x + int(0.20 * w)
    my = y + int(0.62 * h)
    mw = int(0.60 * w)
    mh = int(0.22 * h)
    return mx, my, mw, mh


def _eyes_roi_from_face(face_bbox: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Approximate left/right eye ROIs as fractions of the face bbox.
    Used for blinking overlays (no landmarks dependency).
    """
    x, y, w, h = face_bbox

    # Upper-mid region of face: eyes
    ey = y + int(0.28 * h)
    eh = int(0.14 * h)

    # Left eye region
    lx = x + int(0.18 * w)
    lw = int(0.26 * w)

    # Right eye region
    rx = x + int(0.56 * w)
    rw = int(0.26 * w)

    left = (lx, ey, lw, eh)
    right = (rx, ey, rw, eh)
    return left, right


def _mouth_open_factor(frame_idx: int, fps: int, *, period_seconds: float = 1.0) -> float:
    """
    Smooth mouth opening in [0,1] using a sine wave.
    """
    t = frame_idx / float(fps)
    val = (math.sin(2.0 * math.pi * (t / period_seconds)) + 1.0) / 2.0
    return max(0.0, min(1.0, val))


def _apply_mouth_warp(
    frame_bgr: np.ndarray,
    mouth_roi: Tuple[int, int, int, int],
    open_factor: float,
    *,
    jitter_pixels: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Apply a simple vertical warp to the mouth ROI to simulate opening/closing.
    Optional jitter (for v2): random vertical offset per frame.
    """
    mx, my, mw, mh = mouth_roi
    H, W = frame_bgr.shape[:2]

    mx = max(0, min(mx, W - 1))
    my = max(0, min(my, H - 1))
    mw = max(1, min(mw, W - mx))
    mh = max(1, min(mh, H - my))

    roi = frame_bgr[my:my + mh, mx:mx + mw].copy()

    # Scale vertically: [1.0 .. 1.6]
    scale = 1.0 + 0.6 * open_factor
    new_h = max(1, int(mh * scale))
    warped = cv2.resize(roi, (mw, new_h), interpolation=cv2.INTER_LINEAR)

    # Center paste back into original ROI bounds; allow optional jitter in y
    top = my + (mh - new_h) // 2
    if jitter_pixels > 0 and rng is not None:
        top += int(rng.integers(-jitter_pixels, jitter_pixels + 1))

    bottom = top + new_h

    src_y0 = 0
    src_y1 = new_h

    if top < 0:
        src_y0 = -top
        top = 0
    if bottom > H:
        src_y1 -= (bottom - H)
        bottom = H

    if src_y1 <= src_y0:
        return

    frame_bgr[top:bottom, mx:mx + mw] = warped[src_y0:src_y1]


def _apply_blink_overlay(
    frame_bgr: np.ndarray,
    eye_roi: Tuple[int, int, int, int],
    blink_strength: float,
) -> None:
    """
    Simulate eyelid closure by overlaying a horizontal dark band in the eye ROI.
    blink_strength in [0,1], where 1 is fully closed.
    """
    if blink_strength <= 0.0:
        return

    x, y, w, h = eye_roi
    H, W = frame_bgr.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    # Band thickness increases with blink_strength
    band_h = max(1, int(h * (0.25 + 0.60 * blink_strength)))
    band_y = y + (h - band_h) // 2

    # Blend a dark band (not pure black, to look more natural)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x, band_y), (x + w, band_y + band_h), (15, 15, 15), thickness=-1)

    alpha = 0.65 * blink_strength  # stronger closure -> stronger overlay
    frame_bgr[:] = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


def _blink_strength_periodic(frame_idx: int, fps: int, *, every_seconds: float = 2.0, blink_len_frames: int = 4) -> float:
    """
    Perfectly periodic blink: every N seconds, blink for blink_len_frames.
    Strength ramps up then down within the blink window.
    """
    period_frames = max(1, int(round(every_seconds * fps)))
    pos = frame_idx % period_frames

    if pos >= blink_len_frames:
        return 0.0

    # Triangular ramp: 0->1->0 over blink_len_frames
    mid = (blink_len_frames - 1) / 2.0
    if mid <= 0:
        return 1.0
    strength = 1.0 - abs(pos - mid) / mid
    return float(max(0.0, min(1.0, strength)))


def _blink_strength_randomish(frame_idx: int, fps: int, rng: np.random.Generator) -> float:
    """
    Deterministic pseudo-random blinking:
    - Use RNG to sample occasional blink events
    - Each event lasts a short number of frames
    """
    # Create a low-rate event trigger based on frame index blocks
    # We sample a trigger every ~0.5 seconds
    block = max(1, int(round(0.5 * fps)))
    if frame_idx % block == 0:
        # With small probability, start a blink event in this block
        # (deterministic given rng seed)
        trigger = rng.random() < 0.22  # ~22% chance per block
        if trigger:
            rng.integers(0, 1)  # advance rng slightly (stability)
            # store event in rng state via a small offset
            # We'll encode a "blink start" by putting a marker in a simple dict-like way:
            # (we avoid global state by deriving from rng + frame_idx deterministically)
            pass

    # Deterministic blink strength derived from hashing frame_idx with rng
    # If you want cleaner stateful events later, refactor to a BlinkScheduler class.
    # Here: occasionally produce a short blink burst using a stable pseudo-random gate.
    gate = (rng.integers(0, 10_000_000) + frame_idx) % 97
    if gate < 3:
        # 3-frame blink ramp
        pos = gate  # 0,1,2
        return [0.4, 1.0, 0.5][pos]
    return 0.0


def _apply_lighting_flicker(frame_bgr: np.ndarray, frame_idx: int, fps: int, *, amp: float = 0.12, period_seconds: float = 3.0) -> None:
    """
    Multiply frame brightness by a periodic factor.
    """
    t = frame_idx / float(fps)
    factor = 1.0 + amp * math.sin(2.0 * math.pi * (t / period_seconds))
    out = frame_bgr.astype(np.float32) * factor
    frame_bgr[:] = _clamp_u8(out)


def _apply_head_drift(frame_bgr: np.ndarray, frame_idx: int, fps: int, *, max_pixels: int = 3) -> np.ndarray:
    """
    Apply a tiny smooth translation to the entire frame (simulates slight head/camera drift).
    Uses a deterministic sinus drift.
    """
    t = frame_idx / float(fps)
    dx = int(round(max_pixels * math.sin(2.0 * math.pi * t / 5.0)))
    dy = int(round(max_pixels * math.sin(2.0 * math.pi * t / 7.0)))

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        frame_bgr,
        M,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return shifted


# =========================
# Ground truth descriptions
# =========================

def _ground_truth_text(variant: VideoVariant, *, base_image_path: str, face_bbox: Tuple[int, int, int, int]) -> str:
    x, y, w, h = face_bbox
    header = (
        "This video was synthetically generated from a static real face image.\n"
        f"Base image: {base_image_path}\n"
        f"Detected face bbox: x={x}, y={y}, w={w}, h={h}\n"
    )

    if variant == VideoVariant.V0:
        body = (
            "Variant: v0\n"
            "The mouth shows continuous, smooth motion over time.\n"
            "The eyes remain completely static and do not blink at any point.\n"
            "This inconsistency is intentional and represents an unnatural facial behavior.\n"
        )
    elif variant == VideoVariant.V1:
        body = (
            "Variant: v1\n"
            "The mouth shows continuous, smooth motion over time.\n"
            "Blinking exists but occurs with perfectly periodic timing.\n"
            "This regularity is intentional and represents an unnatural facial behavior.\n"
        )
    elif variant == VideoVariant.V2:
        body = (
            "Variant: v2\n"
            "The mouth motion contains abrupt, frame-to-frame discontinuities (jitter).\n"
            "This jitter is intentional and represents an unnatural temporal artifact.\n"
        )
    elif variant == VideoVariant.V3:
        body = (
            "Variant: v3\n"
            "The scene brightness flickers periodically despite a static scene.\n"
            "This lighting inconsistency is intentional and represents an unnatural artifact.\n"
        )
    elif variant == VideoVariant.V4:
        body = (
            "Variant: v4\n"
            "The mouth shows continuous, smooth motion over time.\n"
            "Blinking occurs in a pseudo-random manner (deterministic given the seed).\n"
            "A tiny smooth drift is applied to simulate natural micro-movement.\n"
            "No single intentional 'fake' artifact is introduced; this is a baseline.\n"
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return header + body


def _write_ground_truth_description(path: str, text: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# Frame rendering
# =========================

def _render_frame(
    base_bgr: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    variant: VideoVariant,
    frame_idx: int,
    spec: VideoGenSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Render one frame for a given variant based on a static base image.
    """
    frame = base_bgr.copy()

    mouth_roi = _mouth_roi_from_face(face_bbox)
    left_eye_roi, right_eye_roi = _eyes_roi_from_face(face_bbox)

    # Mouth motion always present in v0,v1,v2,v4 (and also fine in v3 for realism).
    open_factor = _mouth_open_factor(frame_idx, spec.fps, period_seconds=1.0)

    if variant == VideoVariant.V2:
        # Jittery mouth: add per-frame discontinuity
        _apply_mouth_warp(frame, mouth_roi, open_factor, jitter_pixels=5, rng=rng)
    else:
        _apply_mouth_warp(frame, mouth_roi, open_factor, jitter_pixels=0, rng=None)

    # Eye behavior
    if variant == VideoVariant.V0:
        # No blinking: do nothing
        pass

    elif variant == VideoVariant.V1:
        # Perfectly periodic blink
        strength = _blink_strength_periodic(frame_idx, spec.fps, every_seconds=2.0, blink_len_frames=5)
        _apply_blink_overlay(frame, left_eye_roi, strength)
        _apply_blink_overlay(frame, right_eye_roi, strength)

    elif variant == VideoVariant.V4:
        # Pseudo-random blinking (deterministic given seed)
        strength = _blink_strength_randomish(frame_idx, spec.fps, rng)
        _apply_blink_overlay(frame, left_eye_roi, strength)
        _apply_blink_overlay(frame, right_eye_roi, strength)

    # Lighting behavior
    if variant == VideoVariant.V3:
        _apply_lighting_flicker(frame, frame_idx, spec.fps, amp=0.14, period_seconds=3.0)

    # Head/camera micro drift (baseline realism)
    if variant == VideoVariant.V4:
        frame = _apply_head_drift(frame, frame_idx, spec.fps, max_pixels=3)

    return frame


# =========================
# Public API
# =========================

def generate_video(
    variant: VideoVariant | str,
    out_video_path: str,
    out_description_path: str,
    *,
    base_image_path: str,
    width: int = 512,
    height: int = 512,
    fps: int = 30,
    duration_seconds: int = 10,
    seed: int = 123,
) -> None:
    """
    Generate a synthetic photo-based video variant (v0–v4) and a ground-truth TXT.

    Command line example:
        python src/deepfake_detector/generation/opencv_generator.py \
          --variant v0 \
          --base-image assets/base_faces/face_01.jpg \
          --out-video assets/videos/v0.mp4 \
          --out-desc assets/videos/v0.txt

    Python example:
        from deepfake_detector.generation.opencv_generator import generate_video
        generate_video(
            "v0",
            out_video_path="assets/videos/v0.mp4",
            out_description_path="assets/videos/v0.txt",
            base_image_path="assets/base_faces/face_01.jpg",
        )
    """
    variant_enum = VideoVariant(variant) if not isinstance(variant, VideoVariant) else variant
    spec = VideoGenSpec(width=width, height=height, fps=fps, duration_seconds=duration_seconds, seed=seed)
    _validate_spec(spec)

    if not base_image_path:
        raise ValueError("base_image_path is required for all variants (static face image).")

    _ensure_parent_dir(out_video_path)
    _ensure_parent_dir(out_description_path)

    rng = np.random.default_rng(spec.seed)

    # Load and resize base image to target resolution
    base_bgr = _load_base_image(base_image_path, spec.width, spec.height)

    # Detect face bbox once (stable across frames)
    face_bbox = _detect_face_bbox_bgr(base_bgr)

    # Write ground-truth description
    desc = _ground_truth_text(variant_enum, base_image_path=base_image_path, face_bbox=face_bbox)

    total_frames = spec.fps * spec.duration_seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, spec.fps, (spec.width, spec.height))
    if not writer.isOpened():
        raise OSError(
            "Failed to open VideoWriter. Your OpenCV build may lack MP4 codec support. "
            "Try installing ffmpeg or using a different fourcc."
        )

    try:
        for i in range(total_frames):
            frame = _render_frame(base_bgr, face_bbox, variant_enum, i, spec, rng)
            writer.write(frame)
    finally:
        writer.release()

    _write_ground_truth_description(out_description_path, desc)


# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic photo-based videos for deepfake detection experiments.")
    p.add_argument("--variant", type=str, required=True, choices=[v.value for v in VideoVariant])
    p.add_argument("--base-image", type=str, required=True, help="Path to a static real face image (frontal face).")
    p.add_argument("--out-video", type=str, required=True, help="Output MP4 path.")
    p.add_argument("--out-desc", type=str, required=True, help="Output TXT description path.")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration-seconds", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_video(
        args.variant,
        out_video_path=args.out_video,
        out_description_path=args.out_desc,
        base_image_path=args.base_image,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration_seconds,
        seed=args.seed,
    )
    print(f"Generated video: {args.out_video}")
    print(f"Generated description: {args.out_desc}")
