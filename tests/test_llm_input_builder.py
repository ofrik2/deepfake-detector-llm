import json

import pytest

from src.deepfake_detector.llm_input.input_builder import (
    _read_json,
    build_llm_input,
    save_llm_input,
)


class TestReadJson:
    def test_successful_read(self, tmp_path):
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.json"
        with file_path.open("w") as f:
            json.dump(data, f)

        result = _read_json(str(file_path))
        assert result == data

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="JSON not found"):
            _read_json("nonexistent.json")


class TestBuildLlmInput:
    def test_basic_build(self, tmp_path):
        # Create manifest
        manifest = {
            "frames": [
                {"frame_index": 0, "timestamp_seconds": 0.0, "filename": "frame_000000.jpg"},
                {"frame_index": 5, "timestamp_seconds": 0.2, "filename": "frame_000001.jpg"},
                {"frame_index": 10, "timestamp_seconds": 0.4, "filename": "frame_000002.jpg"},
                {"frame_index": 15, "timestamp_seconds": 0.6, "filename": "frame_000003.jpg"},
            ],
            "sampling_mode": "uniform",
            "fps": 25.0,
            "duration_seconds": 0.6,
            "resize_max": 512,
            "video_path": "/path/to/video.mp4",
        }
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        # Create evidence
        evidence = {
            "roi_method": "haar_face_per_frame",
            "global_motion_mean": 1.5,
            "notes": ["Test note"],
        }
        evidence_path = tmp_path / "evidence.json"
        with evidence_path.open("w") as f:
            json.dump(evidence, f)

        result = build_llm_input(
            manifest_path=str(manifest_path), evidence_path=str(evidence_path), max_keyframes=2
        )

        assert result["video_id"] == "video"
        assert result["sampling"]["sampled_frame_count"] == 4
        assert result["sampling"]["sampling_mode"] == "uniform"
        assert len(result["keyframes"]) == 2
        assert result["evidence"] == evidence

        # Check keyframes selection (should be evenly spaced)
        assert result["keyframes"][0]["sample_pos"] == 0
        assert result["keyframes"][1]["sample_pos"] == 3  # Last one

    def test_custom_video_id(self, tmp_path):
        manifest = {"frames": [{"frame_index": 0, "filename": "f.jpg"}]}
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        evidence = {}
        evidence_path = tmp_path / "evidence.json"
        with evidence_path.open("w") as f:
            json.dump(evidence, f)

        result = build_llm_input(
            manifest_path=str(manifest_path), evidence_path=str(evidence_path), video_id="custom_id"
        )

        assert result["video_id"] == "custom_id"

    def test_no_frames_in_manifest(self, tmp_path):
        manifest = {"frames": []}
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        evidence = {}
        evidence_path = tmp_path / "evidence.json"
        with evidence_path.open("w") as f:
            json.dump(evidence, f)

        with pytest.raises(ValueError, match="Manifest has no frames"):
            build_llm_input(manifest_path=str(manifest_path), evidence_path=str(evidence_path))

    def test_max_keyframes_zero(self, tmp_path):
        manifest = {"frames": [{"frame_index": 0, "filename": "f.jpg"}]}
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        evidence = {}
        evidence_path = tmp_path / "evidence.json"
        with evidence_path.open("w") as f:
            json.dump(evidence, f)

        result = build_llm_input(
            manifest_path=str(manifest_path), evidence_path=str(evidence_path), max_keyframes=0
        )

        assert result["keyframes"] == []

    def test_max_keyframes_more_than_frames(self, tmp_path):
        manifest = {
            "frames": [
                {"frame_index": 0, "filename": "f1.jpg"},
                {"frame_index": 1, "filename": "f2.jpg"},
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        evidence = {}
        evidence_path = tmp_path / "evidence.json"
        with evidence_path.open("w") as f:
            json.dump(evidence, f)

        result = build_llm_input(
            manifest_path=str(manifest_path), evidence_path=str(evidence_path), max_keyframes=5
        )

        assert len(result["keyframes"]) == 2  # All frames

    def test_custom_frames_dir(self, tmp_path):
        manifest = {"frames": [{"frame_index": 0, "filename": "f.jpg"}]}
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        evidence = {}
        evidence_path = tmp_path / "evidence.json"
        with evidence_path.open("w") as f:
            json.dump(evidence, f)

        custom_frames_dir = tmp_path / "custom_frames"
        custom_frames_dir.mkdir()

        result = build_llm_input(
            manifest_path=str(manifest_path),
            evidence_path=str(evidence_path),
            frames_dir=str(custom_frames_dir),
        )

        assert result["keyframes"][0]["path"].startswith(str(custom_frames_dir))

    def test_keyframe_selection_even_spacing(self, tmp_path):
        # Test with 10 frames, max_keyframes=3
        frames = [{"frame_index": i, "filename": f"f{i}.jpg"} for i in range(10)]
        manifest = {"frames": frames}
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        evidence = {}
        evidence_path = tmp_path / "evidence.json"
        with evidence_path.open("w") as f:
            json.dump(evidence, f)

        result = build_llm_input(
            manifest_path=str(manifest_path), evidence_path=str(evidence_path), max_keyframes=3
        )

        positions = [kf["sample_pos"] for kf in result["keyframes"]]
        # Should be 0, 4, 9 (evenly spaced)
        assert positions == [0, 4, 9]


class TestSaveLlmInput:
    def test_save_creates_file(self, tmp_path):
        data = {"test": "data"}
        out_path = tmp_path / "output.json"

        save_llm_input(data, str(out_path))

        assert out_path.exists()
        with out_path.open("r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_creates_directories(self, tmp_path):
        data = {"test": "data"}
        out_path = tmp_path / "subdir" / "output.json"

        save_llm_input(data, str(out_path))

        assert out_path.exists()
        assert out_path.parent.exists()
