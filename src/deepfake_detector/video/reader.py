from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def open_video(video_path: str) -> cv2.VideoCapture:
    """
    Open a video file and return a cv2.VideoCapture.

    Raises:
        FileNotFoundError: if path does not exist
        OSError: if OpenCV cannot open the video
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {video_path}")

    return cap


def read_video_meta(video_path: str, cap: Optional[cv2.VideoCapture] = None) -> VideoMeta:
    """
    Read basic video metadata (fps, frame_count, width, height, duration).
    If cap is provided, it will be used (and not released here).
    """
    own_cap = False
    if cap is None:
        cap = open_video(video_path)
        own_cap = True

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Some codecs report fps=0; we keep duration as 0 in that case.
        duration_seconds = float(frame_count / fps) if (fps and frame_count) else 0.0

        return VideoMeta(
            path=video_path,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
        )
    finally:
        if own_cap:
            cap.release()


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Tuple[bool, Optional[any]]:
    """
    Seek to a frame index and read it.

    Returns:
        (ok, frame_bgr)
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, frame
