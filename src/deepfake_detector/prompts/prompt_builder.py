from __future__ import annotations

from typing import Any, Dict, List


def build_prompt_text(llm_input: Dict[str, Any]) -> str:
    sampling = llm_input.get("sampling", {}) or {}
    evidence = llm_input.get("evidence", {}) or {}
    keyframes = llm_input.get("keyframes", []) or []

    sampled_frame_count = sampling.get("sampled_frame_count")
    sampling_mode = sampling.get("sampling_mode")
    fps = sampling.get("fps")
    duration_seconds = sampling.get("duration_seconds")

    roi_method = evidence.get("roi_method", "unknown")
    notes = evidence.get("notes", []) or []

    lines: List[str] = []
    lines.append(
        "You are analyzing a short video represented by sampled frames and extracted evidence."
    )
    lines.append("")
    lines.append("Context:")
    lines.append(f"- Sampled frames: {sampled_frame_count} (mode={sampling_mode})")
    lines.append(f"- FPS: {fps}, duration_seconds: {duration_seconds}")

    if keyframes:
        ts = [kf.get("timestamp_seconds") for kf in keyframes if "timestamp_seconds" in kf]
        lines.append(f"- Keyframe timestamps (seconds): {ts}")
    else:
        lines.append("- Keyframes: none provided (text-only run)")

    lines.append("")
    lines.append("Extracted evidence (computed from sampled frames):")
    lines.append(f"- ROI method: {roi_method}")

    # ---- Interpretation hints (keep short + consistent) ----
    lines.append("")
    lines.append("How to interpret the evidence (heuristics):")

    lines.append("- Boundary calibration:")
    lines.append(
        "  - High boundary_face_over_bg_mean (e.g., >1.5) can indicate a face mask edge or compositing artifact."
    )
    lines.append(
        "  - Low boundary_face_over_bg_mean (<=1.0) or negative boundary_face_minus_bg_mean is usually NOT suspicious by itself;"
    )
    lines.append(
        "    it often occurs when the background has stronger edges or texture than the face region."
    )

    lines.append("- Blinking behavior:")
    lines.append(
        "  - If blink_dip_fraction is near 0 AND blink_like_events == 0, blinking is likely absent or unnaturally rare."
    )
    lines.append("  - Absence of blinking over many frames can be indicative of manipulation.")

    lines.append("- Motion consistency:")
    lines.append(
        "  - If eyes_over_mouth_motion_ratio is very low AND blinking is minimal, eyes may be unnaturally static relative to mouth motion."
    )
    lines.append(
        "  - Stable face detection + low bbox jitter supports consistency, but does not guarantee REAL."
    )

    # Motion evidence
    if "global_motion_mean" in evidence:
        lines.append(f"- Global motion mean abs diff: {evidence['global_motion_mean']}")
    if "mouth_motion_mean" in evidence:
        lines.append(f"- Mouth-region motion mean abs diff: {evidence['mouth_motion_mean']}")
    if "eyes_motion_mean" in evidence:
        lines.append(f"- Eye-region motion mean abs diff: {evidence['eyes_motion_mean']}")

    # Face stability
    if "face_bbox_none_ratio" in evidence:
        lines.append(f"- Face detection missing ratio: {evidence['face_bbox_none_ratio']}")
        lines.append(
            f"- Face bbox center jitter mean (normalized): {evidence.get('face_bbox_center_jitter_mean')}"
        )
        lines.append(
            f"- Face bbox size jitter mean (normalized): {evidence.get('face_bbox_size_jitter_mean')}"
        )

    # Boundary (calibrated)
    if "boundary_face_ratio_mean" in evidence:
        lines.append(
            f"- Face-boundary edge ratio mean (ring/inner): {evidence['boundary_face_ratio_mean']}"
        )
        lines.append(
            f"- Background-boundary edge ratio mean: {evidence.get('boundary_bg_ratio_mean')}"
        )
        lines.append(f"- Boundary face_over_bg mean: {evidence.get('boundary_face_over_bg_mean')}")
        lines.append(
            f"- Boundary face_minus_bg mean: {evidence.get('boundary_face_minus_bg_mean')}"
        )
    elif "boundary_edge_ratio_mean" in evidence:
        lines.append(
            f"- Face-boundary edge ratio mean (ring/inner): {evidence['boundary_edge_ratio_mean']}"
        )

    # Motion spike summaries
    if "eyes_motion_p95" in evidence:
        lines.append(
            f"- Eyes motion p95: {evidence['eyes_motion_p95']}, max/mean: {evidence.get('eyes_motion_max_over_mean')}"
        )
    if "mouth_motion_p95" in evidence:
        lines.append(
            f"- Mouth motion p95: {evidence['mouth_motion_p95']}, max/mean: {evidence.get('mouth_motion_max_over_mean')}"
        )

    # Blink scarcity features (we’ll add these in basic_signals.py below)
    if "eye_openness_range" in evidence:
        lines.append(f"- Eye openness range (max-min): {evidence['eye_openness_range']}")
        lines.append(f"- Blink dip fraction: {evidence.get('blink_dip_fraction')}")
        lines.append(
            f"- Blink estimated count: {evidence.get('estimated_blink_count')}, confidence: {evidence.get('blink_confidence')}"
        )

    if "blink_like_events" in evidence:
        lines.append(f"- Blink-like events detected: {evidence['blink_like_events']}")

    if notes:
        lines.append("- Notes:")
        for n in notes:
            lines.append(f"  - {n}")

    lines.append("")
    lines.append("Task:")
    lines.append(
        "Decide whether the video is more likely REAL or MANIPULATED based on the evidence and frames."
    )
    lines.append(
        "Explain your reasoning using the evidence. If uncertain, say UNCERTAIN and explain what is missing."
    )
    lines.append("")
    lines.append("Output format (strict):")
    lines.append("Label: REAL|MANIPULATED|UNCERTAIN")
    lines.append("Reason: <1-5 sentences>")

    return "\n".join(lines)
