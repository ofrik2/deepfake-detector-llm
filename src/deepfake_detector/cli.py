from __future__ import annotations

import argparse
import json

from .detectors.registry import discover_plugins, get_detector, list_detectors


def main():
    discover_plugins()
    p = argparse.ArgumentParser(prog="deepfake-detector-llm")
    p.add_argument("--list-detectors", action="store_true", help="List available detectors")
    sub = p.add_subparsers(dest="cmd", required=False)

    detect = sub.add_parser("detect", help="Run detection pipeline on a video")
    detect.add_argument("--video", required=True, help="Path to input video")
    detect.add_argument("--out", required=True, help="Output directory for artifacts")
    detect.add_argument("--detector", default="llm", help="Detector to use")
    detect.add_argument("--llm", default="mock", choices=["mock", "azure"], help="LLM backend")
    detect.add_argument("--num-frames", type=int, default=12)
    detect.add_argument("--max-keyframes", type=int, default=8)

    args = p.parse_args()

    if args.list_detectors:
        print("Available detectors:")
        for name in list_detectors():
            print(f"  - {name}")
        return

    if not args.cmd:
        p.print_help()
        import sys

        sys.exit(0)

    if args.cmd == "detect":
        detector_cls = get_detector(args.detector)
        if not detector_cls:
            print(f"Error: Detector '{args.detector}' not found.")
            print(f"Available detectors: {', '.join(list_detectors())}")
            return

        detector = detector_cls()
        config = {
            "llm_backend": args.llm,
            "num_frames": args.num_frames,
            "max_keyframes": args.max_keyframes,
        }

        result = detector.detect(video_path=args.video, out_dir=args.out, config=config)

        # Convert DetectorResult to dict for printing
        summary = {
            "label": result.label,
            "rationale": result.rationale,
            "evidence_used": result.evidence_used,
            "metadata": result.metadata,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
