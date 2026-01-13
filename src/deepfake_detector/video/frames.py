from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

import cv2

from .reader import VideoMeta, open_video, read_frame_at, read_video_meta


SamplingMode = Literal["uniform", "every_n"]


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp_seconds: float
    filename: str


@dataclass(frozen=True)
class FramesManifest:
    video_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    sampling_mode: SamplingMode
    num_frames_requested: int
    every_n_requested: int
    resize_max: int
    frames: List[FrameRecord]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resize_keep_aspect(frame_bgr, resize_max: int):
    """
    Resize frame so that max(width,height) == resize_max, preserving aspect ratio.
    If resize_max <= 0, or already smaller, return original.
    """
    if resize_max <= 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= resize_max:
        return frame_bgr

    scale = resize_max / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _uniform_indices(frame_count: int, num_frames: int) -> List[int]:
    """
    Deterministically select num_frames indices uniformly across [0..frame_count-1].
    """
    if frame_count <= 0:
        return []
    if num_frames <= 0:
        return []
    if num_frames >= frame_count:
        return list(range(frame_count))

    # Use evenly spaced positions including endpoints.
    # Example: frame_count=100, num_frames=5 -> [0, 24, 49, 74, 99]
    idxs = []
    for i in range(num_frames):
        pos = round(i * (frame_count - 1) / (num_frames - 1))
        idxs.append(int(pos))
    # Deduplicate in case of rounding collisions
    idxs = sorted(set(idxs))
    return idxs


def _every_n_indices(frame_count: int, every_n: int, max_frames: int) -> List[int]:
    if frame_count <= 0 or every_n <= 0:
        return []
    idxs = list(range(0, frame_count, every_n))
    if max_frames > 0:
        idxs = idxs[:max_frames]
    return idxs


def extract_frames(
    video_path: str,
    out_dir: str,
    *,
    mode: SamplingMode = "uniform",
    num_frames: int = 12,
    every_n: int = 10,
    max_frames: int = 120,
    resize_max: int = 512,
    jpeg_quality: int = 90,
    write_manifest: bool = True,
) -> Dict:
    """
    Extract frames deterministically from a video and write them to out_dir.

    Writes:
      out_dir/
        frame_000000.jpg
        frame_000001.jpg
        ...
        frames_manifest.json

    Returns:
      A dict version of FramesManifest.

    Parameters
    ----------
    mode:
      - "uniform": pick exactly num_frames uniformly across the whole video.
      - "every_n": pick frames every N frames (capped by max_frames).
    num_frames:
      Used when mode="uniform".
    every_n:
      Used when mode="every_n".
    max_frames:
      Cap for mode="every_n" to prevent too many frames.
    resize_max:
      Resize frames so max(width,height) <= resize_max (keeps aspect ratio).
    jpeg_quality:
      0-100. Higher = better quality, larger files.
    """
    _ensure_dir(out_dir)

    cap = open_video(video_path)
    try:
        meta: VideoMeta = read_video_meta(video_path, cap=cap)
        if meta.frame_count <= 0:
            raise ValueError(f"Video has no frames (frame_count={meta.frame_count}).")

        if mode == "uniform":
            indices = _uniform_indices(meta.frame_count, num_frames)
        elif mode == "every_n":
            indices = _every_n_indices(meta.frame_count, every_n, max_frames)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

        if not indices:
            raise ValueError("No frames selected. Check sampling parameters.")

        frames: List[FrameRecord] = []

        for j, frame_idx in enumerate(indices):
            ok, frame = read_frame_at(cap, frame_idx)
            if not ok or frame is None:
                # Skip unreadable frame; continue to keep robustness
                continue

            frame = _resize_keep_aspect(frame, resize_max)

            # Timestamp is best-effort (requires valid fps)
            ts = float(frame_idx / meta.fps) if meta.fps else 0.0

            filename = f"frame_{j:06d}.jpg"
            out_path = os.path.join(out_dir, filename)

            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(0, min(100, jpeg_quality)))]
            ok_write = cv2.imwrite(out_path, frame, encode_params)
            if not ok_write:
                raise OSError(f"Failed to write frame to: {out_path}")

            frames.append(FrameRecord(frame_index=int(frame_idx), timestamp_seconds=ts, filename=filename))

        if len(frames) == 0:
            raise ValueError("Failed to extract any frames (all reads failed).")

        manifest = FramesManifest(
            video_path=meta.path,
            fps=meta.fps,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
            duration_seconds=meta.duration_seconds,
            sampling_mode=mode,
            num_frames_requested=int(num_frames),
            every_n_requested=int(every_n),
            resize_max=int(resize_max),
            frames=frames,
        )

        manifest_dict = asdict(manifest)

        if write_manifest:
            manifest_path = os.path.join(out_dir, "frames_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_dict, f, indent=2)

        return manifest_dict

    finally:
        cap.release()
