from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMInput:
    """
    A compact, JSON-serializable payload the LLM can reason over.
    This is the single source of truth for what the model is told.
    """
    video_id: str
    sampling: Dict[str, Any]
    evidence: Dict[str, Any]
    keyframes: List[Dict[str, Any]]  # {sample_pos, frame_index, timestamp_seconds, filename, path}


def _read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_llm_input(
    *,
    manifest_path: str,
    evidence_path: str,
    frames_dir: Optional[str] = None,
    max_keyframes: int = 8,
    video_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a stable LLM input bundle from frames_manifest + evidence JSON.

    - Keyframes are selected evenly across the sampled frames list.
    - frames_dir defaults to the manifest directory (same folder).
    """
    manifest = _read_json(manifest_path)
    evidence = _read_json(evidence_path)

    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError("Manifest has no frames.")

    manifest_dir = Path(manifest_path).resolve().parent
    frames_dir_resolved = Path(frames_dir).resolve() if frames_dir else manifest_dir

    if video_id is None:
        vp = manifest.get("video_path") or "video"
        video_id = Path(vp).stem

    sampling = {
        "sampled_frame_count": len(frames),
        "sampling_mode": manifest.get("sampling_mode"),
        "fps": manifest.get("fps"),
        "duration_seconds": manifest.get("duration_seconds"),
        "sampled_indices": [f.get("frame_index") for f in frames],
        "sampled_timestamps_seconds": [f.get("timestamp_seconds") for f in frames],
        "resize_max": manifest.get("resize_max"),
    }

    # Choose keyframes among the sampled frames
    n = len(frames)
    k = min(max_keyframes, n) if max_keyframes > 0 else 0

    if k == 0:
        chosen_positions: List[int] = []
    elif k == n:
        chosen_positions = list(range(n))
    else:
        chosen_positions = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})

    keyframes: List[Dict[str, Any]] = []
    for pos in chosen_positions:
        f = frames[pos]
        filename = f["filename"]
        keyframes.append(
            {
                "sample_pos": int(pos),
                "frame_index": f.get("frame_index"),
                "timestamp_seconds": f.get("timestamp_seconds"),
                "filename": filename,
                "path": str((frames_dir_resolved / filename).resolve()),
            }
        )

    payload = LLMInput(
        video_id=video_id,
        sampling=sampling,
        evidence=evidence,
        keyframes=keyframes,
    )
    return asdict(payload)


def save_llm_input(llm_input: Dict[str, Any], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2)
