from __future__ import annotations

from typing import Any, Dict, List


def build_prompt_text(llm_input: Dict[str, Any]) -> str:
    """
    Convert an LLMInput dict into a clean prompt text.

    The caller should attach the images referenced in llm_input["keyframes"]
    when using a multimodal model.
    """
    sampling = llm_input.get("sampling", {})
    evidence = llm_input.get("evidence", {})
    keyframes = llm_input.get("keyframes", [])

    keyframe_times = [kf.get("timestamp_seconds") for kf in keyframes]

    roi_method = evidence.get("roi_method", "unknown")
    notes = evidence.get("notes", [])

    lines: List[str] = []
    lines.append("You are analyzing a short video represented by sampled frames and extracted evidence.")
    lines.append("")
    lines.append("Context:")
    lines.append(f"- Sampled frames: {sampling.get('sampled_frame_count')} (mode={sampling.get('sampling_mode')})")
    lines.append(f"- FPS: {sampling.get('fps')}, duration_seconds: {sampling.get('duration_seconds')}")
    lines.append(f"- Keyframe timestamps (seconds): {keyframe_times}")
    lines.append("")
    lines.append("Extracted evidence (computed from sampled frames):")
    lines.append(f"- ROI method: {roi_method}")

    # Motion evidence
    if "global_motion_mean" in evidence:
        lines.append(f"- Global motion mean abs diff: {evidence.get('global_motion_mean')}")
    if "mouth_motion_mean" in evidence:
        lines.append(f"- Mouth-region motion mean abs diff: {evidence.get('mouth_motion_mean')}")
    if "eyes_motion_mean" in evidence:
        lines.append(f"- Eye-region motion mean abs diff: {evidence.get('eyes_motion_mean')}")

    if notes:
        lines.append("- Notes:")
        for n in notes:
            lines.append(f"  - {n}")

    lines.append("")
    lines.append("Task:")
    lines.append("Decide whether the video is more likely REAL or MANIPULATED based on the evidence and frames.")
    lines.append("Explain your reasoning using the evidence. If uncertain, say UNCERTAIN and explain what is missing.")
    lines.append("")
    lines.append("Output format (strict):")
    lines.append("Label: REAL|MANIPULATED|UNCERTAIN")
    lines.append("Reason: <1-5 sentences>")

    return "\n".join(lines)
