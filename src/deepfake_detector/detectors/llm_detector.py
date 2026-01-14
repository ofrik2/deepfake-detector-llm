from __future__ import annotations

from typing import Any, Dict, Optional

from ..pipeline import run_pipeline
from .base import BaseDetector, DetectorResult
from .registry import register_detector


@register_detector("llm")
class LLMDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "llm"

    def detect(
        self, video_path: str, out_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> DetectorResult:
        config = config or {}
        llm_backend = config.get("llm_backend", "mock")
        num_frames = config.get("num_frames", 12)
        max_keyframes = config.get("max_keyframes", 8)

        # Call the existing run_pipeline logic
        # Note: run_pipeline already saves artifacts and returns a summary
        summary = run_pipeline(
            video_path=video_path,
            out_dir=out_dir,
            llm_backend=llm_backend,
            num_frames=num_frames,
            max_keyframes=max_keyframes,
        )

        return DetectorResult(
            label=summary["label"],
            rationale=summary["reason"],
            evidence_used=[summary["evidence_path"], summary["prompt_path"]],
            metadata={
                "llm_backend": llm_backend,
                "num_frames": num_frames,
                "max_keyframes": max_keyframes,
                "out_dir": summary["out_dir"],
            },
        )
