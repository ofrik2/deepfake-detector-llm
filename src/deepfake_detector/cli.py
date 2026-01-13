from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def main():
    p = argparse.ArgumentParser(prog="deepfake-detector-llm")
    sub = p.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect", help="Run detection pipeline on a video")
    detect.add_argument("--video", required=True, help="Path to input video")
    detect.add_argument("--out", required=True, help="Output directory for artifacts")
    detect.add_argument("--llm", default="mock", choices=["mock", "azure"], help="LLM backend")
    detect.add_argument("--num-frames", type=int, default=12)
    detect.add_argument("--max-keyframes", type=int, default=8)

    args = p.parse_args()

    if args.cmd == "detect":
        summary = run_pipeline(
            video_path=args.video,
            out_dir=args.out,
            llm_backend=args.llm,
            num_frames=args.num_frames,
            max_keyframes=args.max_keyframes,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
