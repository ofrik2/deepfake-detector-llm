from pathlib import Path

from src.deepfake_detector.evidence.basic_signals import compute_basic_signals, save_basic_signals

REPO_ROOT = Path(__file__).resolve().parents[1]
manifest_path = REPO_ROOT / "assets" / "frames" / "man_speak_blink" / "frames_manifest.json"
out_path = REPO_ROOT / "assets" / "frames" / "man_speak_blink" / "evidence_basic.json"

evidence = compute_basic_signals(str(manifest_path))

print("roi_method:", evidence["roi_method"])
print("global_motion_mean:", evidence["global_motion_mean"])
print("mouth_motion_mean:", evidence["mouth_motion_mean"])
print("eyes_motion_mean:", evidence["eyes_motion_mean"])
print("notes:", evidence["notes"])

save_basic_signals(evidence, str(out_path))
print("Saved:", out_path)
