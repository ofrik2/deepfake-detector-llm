"""
test_evidence_basic_signals.py

Unit tests for the basic signals evidence computation.
Tests the complex evidence computation with mocked OpenCV and file operations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from deepfake_detector.evidence.basic_signals import (
    compute_basic_signals,
    save_basic_signals,
    EvidenceResult
)


class TestComputeBasicSignals:
    """Test the compute_basic_signals function."""

    @pytest.fixture
    def mock_manifest(self):
        """Mock frames manifest data."""
        return {
            "video_path": "/fake/video.mp4",
            "fps": 30.0,
            "frame_count": 900,
            "width": 1920,
            "height": 1080,
            "duration_seconds": 30.0,
            "sampling_mode": "uniform",
            "num_frames_requested": 12,
            "every_n_requested": 75,
            "resize_max": 512,
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "filename": "frame_0000.jpg"
                },
                {
                    "frame_index": 75,
                    "timestamp_seconds": 2.5,
                    "filename": "frame_0075.jpg"
                },
                {
                    "frame_index": 150,
                    "timestamp_seconds": 5.0,
                    "filename": "frame_0150.jpg"
                }
            ]
        }

    @pytest.fixture
    def mock_frames(self):
        """Mock frame images as numpy arrays."""
        # Create mock 1080x1920 RGB frames
        frame1 = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        frame3 = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        return [frame1, frame2, frame3]

    @pytest.fixture
    def mock_blink_result(self):
        """Mock blink detection result."""
        return {
            "blink_detected": True,
            "estimated_blink_count": 1,
            "blink_confidence": 0.8,
            "eye_openness_series": [0.9, 0.2, 0.85],
            "openness_threshold": 0.3,
            "method": "dip_detection"
        }

    def create_mock_manifest_file(self, manifest_data):
        """Helper to create a temporary manifest file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest_data, f)
            return f.name

    def create_mock_frame_files(self, frames_dir, frame_files, mock_frames):
        """Helper to mock frame files in directory."""
        for i, filename in enumerate(frame_files):
            frame_path = Path(frames_dir) / filename
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            # In real scenario, these would be actual image files
            # For mocking, we'll just ensure the path exists

    @patch('deepfake_detector.evidence.basic_signals.cv2.imread')
    @patch('deepfake_detector.evidence.basic_signals._detect_face_bbox_haar')
    @patch('deepfake_detector.evidence.basic_signals.compute_blink_evidence_from_eyes_roi_series')
    @patch('deepfake_detector.evidence.basic_signals._boundary_edge_ratio_for_roi')
    @patch('deepfake_detector.evidence.basic_signals._background_roi')
    @patch('deepfake_detector.evidence.basic_signals._bbox_jitter_stats')
    @patch('deepfake_detector.evidence.basic_signals._mean_abs_diff')
    @patch('deepfake_detector.evidence.basic_signals._mean_abs_diff_allow_mismatch')
    @patch('deepfake_detector.evidence.basic_signals._crop_roi')
    def test_compute_basic_signals_success_with_faces(
        self,
        mock_crop_roi,
        mock_mean_abs_diff_allow,
        mock_mean_abs_diff,
        mock_bbox_jitter,
        mock_background_roi,
        mock_boundary_ratio,
        mock_blink_compute,
        mock_detect_face,
        mock_imread,
        mock_manifest,
        mock_frames,
        mock_blink_result
    ):
        """Test successful computation with face detection."""
        # Setup mocks
        mock_imread.side_effect = mock_frames
        mock_detect_face.side_effect = [
            (100, 100, 200, 200),  # face detected
            (105, 95, 195, 205),   # face detected
            None  # face not detected, should use fallback
        ]
        mock_blink_compute.return_value = mock_blink_result
        mock_boundary_ratio.return_value = 1.2
        mock_background_roi.return_value = {"x": 0, "y": 0, "w": 100, "h": 100}
        mock_bbox_jitter.return_value = {
            "face_bbox_none_ratio": 0.0,
            "face_bbox_center_jitter_mean": 5.2,
            "face_bbox_size_jitter_mean": 3.1
        }
        mock_mean_abs_diff.side_effect = [12.5, 15.0]  # global motion
        mock_mean_abs_diff_allow.side_effect = [14.2, 16.0, 5.8, 7.0]  # mouth and eyes motion
        mock_crop_roi.return_value = np.ones((32, 64), dtype=np.float32)  # mock cropped ROI

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create manifest file
            manifest_path = self.create_mock_manifest_file(mock_manifest)
            frames_dir = temp_dir

            # Create mock frame files
            self.create_mock_frame_files(frames_dir, ["frame_0000.jpg", "frame_0075.jpg", "frame_0150.jpg"], mock_frames)

            result = compute_basic_signals(manifest_path, frames_dir=frames_dir)

            # Verify result structure
            assert isinstance(result, dict)
            assert result["manifest_path"] == manifest_path
            assert result["frames_dir"] == frames_dir
            assert result["sampled_frame_count"] == 3
            assert result["sampled_frame_files"] == ["frame_0000.jpg", "frame_0075.jpg", "frame_0150.jpg"]
            assert result["roi_method"] == "haar_face_per_frame"
            assert result["face_bbox"] is not None
            assert result["face_bbox"]["x"] == 100
            assert result["face_bbox"]["y"] == 100
            assert result["face_bbox"]["w"] == 200
            assert result["face_bbox"]["h"] == 200

            # Verify motion calculations
            assert result["global_motion_mean"] == 13.75  # (12.5 + 15.0) / 2
            assert result["global_motion_min"] == 12.5
            assert result["global_motion_max"] == 15.0
            assert result["mouth_motion_mean"] == 10.0  # (14.2 + 5.8) / 2 (mock interleaved)
            assert result["eyes_motion_mean"] == 11.5  # (16.0 + 7.0) / 2 (mock interleaved)

            # Verify blink evidence
            assert result["blink_detected"] is True
            assert result["estimated_blink_count"] == 1
            assert result["blink_confidence"] == 0.8
            assert result["eye_openness_series"] == [0.9, 0.2, 0.85]

            # Verify boundary evidence
            assert result["boundary_face_ratio_mean"] > 0
            assert result["boundary_bg_ratio_mean"] > 0
            assert result["boundary_face_over_bg_mean"] > 0

            # Verify notes contain useful information
            assert isinstance(result["notes"], list)
            assert len(result["notes"]) > 0

    @patch('deepfake_detector.evidence.basic_signals.cv2.imread')
    @patch('deepfake_detector.evidence.basic_signals._detect_face_bbox_haar')
    @patch('deepfake_detector.evidence.basic_signals.compute_blink_evidence_from_eyes_roi_series')
    def test_compute_basic_signals_fallback_when_no_faces(
        self,
        mock_blink_compute,
        mock_detect_face,
        mock_imread,
        mock_manifest,
        mock_frames,
        mock_blink_result
    ):
        """Test computation falls back to center-based ROIs when no faces detected."""
        # Setup mocks
        mock_imread.side_effect = mock_frames
        mock_detect_face.return_value = None  # No faces detected
        mock_blink_compute.return_value = mock_blink_result

        with patch('deepfake_detector.evidence.basic_signals._boundary_edge_ratio_for_roi', return_value=1.0), \
             patch('deepfake_detector.evidence.basic_signals._background_roi', return_value={"x": 0, "y": 0, "w": 100, "h": 100}), \
             patch('deepfake_detector.evidence.basic_signals._bbox_jitter_stats', return_value={
                 "face_bbox_none_ratio": 1.0,
                 "face_bbox_center_jitter_mean": 0.0,
                 "face_bbox_size_jitter_mean": 0.0
             }), \
             patch('deepfake_detector.evidence.basic_signals._mean_abs_diff', return_value=10.0), \
             patch('deepfake_detector.evidence.basic_signals._mean_abs_diff_allow_mismatch', return_value=8.0), \
             patch('deepfake_detector.evidence.basic_signals._crop_roi', return_value=np.ones((32, 64), dtype=np.float32)):

            with tempfile.TemporaryDirectory() as temp_dir:
                manifest_path = self.create_mock_manifest_file(mock_manifest)
                frames_dir = temp_dir

                result = compute_basic_signals(manifest_path, frames_dir=frames_dir)

                # Verify fallback behavior
                assert result["roi_method"] == "fallback_center"
                assert result["face_bbox"] is None
                assert "Face detection failed on all frames" in str(result["notes"])

    def test_compute_basic_signals_empty_manifest(self):
        """Test that empty manifest raises ValueError."""
        empty_manifest = {"frames": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self.create_mock_manifest_file(empty_manifest)

            with pytest.raises(ValueError, match="Manifest contains no frames"):
                compute_basic_signals(manifest_path)

    def test_compute_basic_signals_missing_manifest_file(self):
        """Test that missing manifest file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_basic_signals("/nonexistent/manifest.json")

    @patch('deepfake_detector.evidence.basic_signals.cv2.imread')
    def test_compute_basic_signals_missing_frame_file(self, mock_imread):
        """Test that missing frame file raises FileNotFoundError."""
        mock_imread.return_value = None  # cv2.imread returns None for missing files

        manifest = {
            "frames": [{"filename": "missing.jpg"}]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self.create_mock_manifest_file(manifest)

            with pytest.raises(FileNotFoundError, match="Failed to read frame image"):
                compute_basic_signals(manifest_path, frames_dir=temp_dir)

    @patch('deepfake_detector.evidence.basic_signals.cv2.imread')
    @patch('deepfake_detector.evidence.basic_signals._detect_face_bbox_haar')
    @patch('deepfake_detector.evidence.basic_signals.compute_blink_evidence_from_eyes_roi_series')
    def test_compute_basic_signals_single_frame(
        self,
        mock_blink_compute,
        mock_detect_face,
        mock_imread,
        mock_manifest,
        mock_blink_result
    ):
        """Test computation with single frame (edge case)."""
        # Modify manifest to have only one frame
        single_frame_manifest = mock_manifest.copy()
        single_frame_manifest["frames"] = [mock_manifest["frames"][0]]

        mock_imread.return_value = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        mock_detect_face.return_value = (100, 100, 200, 200)
        mock_blink_compute.return_value = mock_blink_result

        with patch('deepfake_detector.evidence.basic_signals._boundary_edge_ratio_for_roi', return_value=1.0), \
             patch('deepfake_detector.evidence.basic_signals._background_roi', return_value={"x": 0, "y": 0, "w": 100, "h": 100}), \
             patch('deepfake_detector.evidence.basic_signals._bbox_jitter_stats', return_value={
                 "face_bbox_none_ratio": 0.0,
                 "face_bbox_center_jitter_mean": 0.0,
                 "face_bbox_size_jitter_mean": 0.0
             }), \
             patch('deepfake_detector.evidence.basic_signals._crop_roi', return_value=np.ones((32, 64), dtype=np.float32)):

            with tempfile.TemporaryDirectory() as temp_dir:
                manifest_path = self.create_mock_manifest_file(single_frame_manifest)

                result = compute_basic_signals(manifest_path, frames_dir=temp_dir)

                # With single frame, motion calculations should be empty/zero
                assert result["sampled_frame_count"] == 1
                assert result["global_per_pair_motion"] == []
                assert result["mouth_per_pair_motion"] == []
                assert result["eyes_per_pair_motion"] == []
                assert result["global_motion_mean"] == 0.0
                assert result["mouth_motion_mean"] == 0.0
                assert result["eyes_motion_mean"] == 0.0


class TestSaveBasicSignals:
    """Test the save_basic_signals function."""

    def test_save_basic_signals_success(self):
        """Test successful saving of evidence to JSON file."""
        evidence = {
            "manifest_path": "/fake/manifest.json",
            "sampled_frame_count": 12,
            "global_motion_mean": 15.5,
            "notes": ["Test note"]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "evidence.json"

            save_basic_signals(evidence, str(out_path))

            # Verify file was created and contains correct data
            assert out_path.exists()
            with open(out_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data == evidence

    def test_save_basic_signals_creates_directory(self):
        """Test that save creates necessary directories."""
        evidence = {"test": "data"}

        with tempfile.TemporaryDirectory() as temp_base:
            out_path = Path(temp_base) / "nested" / "dir" / "evidence.json"

            # Directory shouldn't exist yet
            assert not out_path.parent.exists()

            save_basic_signals(evidence, str(out_path))

            # Directory should now exist
            assert out_path.parent.exists()
            assert out_path.exists()

    def test_save_basic_signals_overwrites_existing(self):
        """Test that save overwrites existing file."""
        evidence1 = {"version": 1}
        evidence2 = {"version": 2}

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "evidence.json"

            # Save first version
            save_basic_signals(evidence1, str(out_path))
            with open(out_path, 'r') as f:
                assert json.load(f)["version"] == 1

            # Save second version (should overwrite)
            save_basic_signals(evidence2, str(out_path))
            with open(out_path, 'r') as f:
                assert json.load(f)["version"] == 2


class TestEvidenceResult:
    """Test the EvidenceResult dataclass."""

    def test_evidence_result_creation(self):
        """Test EvidenceResult can be created with valid data."""
        result = EvidenceResult(
            manifest_path="/fake/manifest.json",
            frames_dir="/fake/frames",
            sampled_frame_count=12,
            sampled_frame_files=["frame_001.jpg", "frame_002.jpg"],
            roi_method="haar_face_per_frame",
            face_bbox={"x": 100, "y": 100, "w": 200, "h": 200},
            face_bbox_series=[{"x": 100, "y": 100, "w": 200, "h": 200}],
            mouth_roi={"x": 120, "y": 180, "w": 60, "h": 40},
            eyes_roi={"x": 120, "y": 120, "w": 60, "h": 40},
            face_bbox_none_ratio=0.0,
            face_bbox_center_jitter_mean=5.2,
            face_bbox_size_jitter_mean=3.1,
            global_motion_mean=12.5,
            global_motion_min=8.0,
            global_motion_max=18.0,
            global_motion_p95=16.0,
            global_per_pair_motion=[10.0, 15.0],
            mouth_motion_mean=14.2,
            mouth_motion_min=10.0,
            mouth_motion_max=20.0,
            mouth_motion_p95=18.0,
            mouth_motion_max_over_mean=1.4,
            mouth_per_pair_motion=[12.0, 16.0],
            eyes_motion_mean=5.8,
            eyes_motion_min=3.0,
            eyes_motion_max=10.0,
            eyes_motion_p95=9.0,
            eyes_motion_max_over_mean=1.7,
            eyes_per_pair_motion=[4.0, 7.0],
            blink_like_events=2,
            blink_detected=True,
            estimated_blink_count=2,
            blink_confidence=0.8,
            eye_openness_series=[0.9, 0.8, 0.1, 0.9],
            openness_threshold=0.3,
            blink_method="dip_detection",
            eye_openness_range=0.8,
            blink_dip_fraction=0.25,
            boundary_face_ratio_mean=1.2,
            boundary_face_ratio_std=0.1,
            boundary_face_ratio_series=[1.1, 1.3],
            boundary_method="laplacian_var_ring_over_inner",
            boundary_bg_ratio_mean=0.8,
            boundary_bg_ratio_std=0.05,
            boundary_face_over_bg_mean=1.5,
            boundary_face_minus_bg_mean=0.4,
            notes=["Test note"]
        )

        assert result.manifest_path == "/fake/manifest.json"
        assert result.sampled_frame_count == 12
        assert result.roi_method == "haar_face_per_frame"
        assert result.global_motion_mean == 12.5
        assert result.blink_detected is True
        assert result.notes == ["Test note"]

    def test_evidence_result_asdict(self):
        """Test EvidenceResult.asdict() conversion."""
        result = EvidenceResult(
            manifest_path="/test.json",
            frames_dir="/frames",
            sampled_frame_count=2,
            sampled_frame_files=["f1.jpg", "f2.jpg"],
            roi_method="test",
            face_bbox=None,
            face_bbox_series=[None, None],
            mouth_roi={"x": 0, "y": 0, "w": 10, "h": 10},
            eyes_roi={"x": 0, "y": 0, "w": 10, "h": 10},
            face_bbox_none_ratio=1.0,
            face_bbox_center_jitter_mean=0.0,
            face_bbox_size_jitter_mean=0.0,
            global_motion_mean=0.0,
            global_motion_min=0.0,
            global_motion_max=0.0,
            global_motion_p95=0.0,
            global_per_pair_motion=[],
            mouth_motion_mean=0.0,
            mouth_motion_min=0.0,
            mouth_motion_max=0.0,
            mouth_motion_p95=0.0,
            mouth_motion_max_over_mean=0.0,
            mouth_per_pair_motion=[],
            eyes_motion_mean=0.0,
            eyes_motion_min=0.0,
            eyes_motion_max=0.0,
            eyes_motion_p95=0.0,
            eyes_motion_max_over_mean=0.0,
            eyes_per_pair_motion=[],
            blink_like_events=0,
            blink_detected=False,
            estimated_blink_count=0,
            blink_confidence=0.0,
            eye_openness_series=[],
            openness_threshold=0.0,
            blink_method="none",
            eye_openness_range=0.0,
            blink_dip_fraction=0.0,
            boundary_face_ratio_mean=0.0,
            boundary_face_ratio_std=0.0,
            boundary_face_ratio_series=[],
            boundary_method="none",
            boundary_bg_ratio_mean=0.0,
            boundary_bg_ratio_std=0.0,
            boundary_face_over_bg_mean=0.0,
            boundary_face_minus_bg_mean=0.0,
            notes=[]
        )

        result_dict = result.__dict__  # asdict would work too
        assert isinstance(result_dict, dict)
        assert result_dict["manifest_path"] == "/test.json"
        assert result_dict["sampled_frame_count"] == 2