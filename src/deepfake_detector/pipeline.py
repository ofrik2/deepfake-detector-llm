from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from .video.frames import extract_frames
from .evidence.basic_signals import compute_basic_signals, save_basic_signals
from .llm_input.input_builder import build_llm_input, save_llm_input
from .prompts.prompt_builder import build_prompt_text
from .llm.mock_client import MockLLMClient
from .decision.parser import parse_llm_output


def run_pipeline(
    *,
    video_path: str,
    out_dir: str,
    llm_backend: str = "mock",   # "mock" for now
    num_frames: int = 12,
    max_keyframes: int = 8,
) -> Dict:
    """
    End-to-end pipeline:
      video -> frames -> evidence -> llm_input -> prompt -> llm -> decision

    Saves artifacts under out_dir and returns a summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Frames
    frames_dir = out / "frames"
    manifest = extract_frames(
        video_path,
        str(frames_dir),
        mode="uniform",
        num_frames=num_frames,
        resize_max=512,
        write_manifest=True,
    )
    manifest_path = frames_dir / "frames_manifest.json"

    # 2) Evidence
    evidence = compute_basic_signals(str(manifest_path))
    evidence_path = frames_dir / "evidence_basic.json"
    save_basic_signals(evidence, str(evidence_path))

    # 3) LLM input
    llm_input = build_llm_input(
        manifest_path=str(manifest_path),
        evidence_path=str(evidence_path),
        frames_dir=str(frames_dir),
        max_keyframes=max_keyframes,
    )
    llm_input_path = out / "llm_input.json"
    save_llm_input(llm_input, str(llm_input_path))

    # 4) Prompt
    prompt_text = build_prompt_text(llm_input)
    prompt_path = out / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # 5) LLM call (using ThreadPoolExecutor for I/O bound task as per Chapter 16)
    image_paths = [kf["path"] for kf in llm_input.get("keyframes", [])]

    if llm_backend == "mock":
        client = MockLLMClient()
    elif llm_backend == "azure":
        from .llm.azure_client import AzureOpenAIClient
        client = AzureOpenAIClient()
    else:
        raise ValueError(f"Unsupported llm_backend: {llm_backend}")

    # Even for a single call, we use ThreadPoolExecutor to satisfy architecture requirements
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.generate, prompt=prompt_text, image_paths=image_paths)
        resp = future.result()

    llm_output_path = out / "llm_output.txt"
    llm_output_path.write_text(resp.raw_text, encoding="utf-8")

    # 6) Decision parse
    decision = parse_llm_output(resp.raw_text)
    decision_path = out / "decision.json"
    decision_path.write_text(
        json.dumps(
            {"label": decision.label, "reason": decision.reason, "raw_text": decision.raw_text},
            indent=2
        ),
        encoding="utf-8",
    )

    return {
        "video_path": video_path,
        "out_dir": str(out),
        "manifest_path": str(manifest_path),
        "evidence_path": str(evidence_path),
        "llm_input_path": str(llm_input_path),
        "prompt_path": str(prompt_path),
        "llm_output_path": str(llm_output_path),
        "decision_path": str(decision_path),
        "label": decision.label,
        "reason": decision.reason,
    }
