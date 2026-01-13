from pathlib import Path

from src.deepfake_detector.video.frames import extract_frames
from src.deepfake_detector.video.quality_checks import run_quality_checks

REPO_ROOT = Path(__file__).resolve().parents[1]
video_path = REPO_ROOT / "assets" / "videos" / "man_speak_blink.mp4"
out_dir = REPO_ROOT / "assets" / "frames" / "man_speak_blink"

print(run_quality_checks(str(video_path)))
manifest = extract_frames(str(video_path), str(out_dir), num_frames=12, mode="uniform")
print(manifest["frames"][:2])
