"""
test_pipeline.py

Unit tests for the main pipeline functionality.
Tests the end-to-end pipeline with mocked dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepfake_detector.decision.parser import Decision
from deepfake_detector.pipeline import run_pipeline


class TestRunPipeline:
    """Test the main run_pipeline function."""

    mock_manifest_data = {
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
            {"frame_index": 0, "timestamp_seconds": 0.0, "filename": "frame_0000.jpg"},
            {"frame_index": 75, "timestamp_seconds": 2.5, "filename": "frame_0075.jpg"},
        ],
    }

    mock_evidence_data = {
        "manifest_path": "/fake/manifest.json",
        "frames_dir": "/fake/frames",
        "sampled_frame_count": 12,
        "sampled_frame_files": ["frame_0000.jpg", "frame_0075.jpg"],
        "roi_method": "haar_face_per_frame",
        "face_bbox": {"x": 100, "y": 100, "w": 200, "h": 200},
        "face_bbox_series": [{"x": 100, "y": 100, "w": 200, "h": 200}],
        "mouth_roi": {"x": 120, "y": 180, "w": 60, "h": 40},
        "eyes_roi": {"x": 120, "y": 120, "w": 60, "h": 40},
        "face_bbox_none_ratio": 0.0,
        "face_bbox_center_jitter_mean": 5.2,
        "face_bbox_size_jitter_mean": 3.1,
        "global_motion_mean": 12.5,
        "global_motion_min": 8.0,
        "global_motion_max": 18.0,
        "global_motion_p95": 16.0,
        "global_per_pair_motion": [10.0, 15.0],
        "mouth_motion_mean": 14.2,
        "mouth_motion_min": 10.0,
        "mouth_motion_max": 20.0,
        "mouth_motion_p95": 18.0,
        "mouth_motion_max_over_mean": 1.4,
        "mouth_per_pair_motion": [12.0, 16.0],
        "eyes_motion_mean": 5.8,
        "eyes_motion_min": 3.0,
        "eyes_motion_max": 10.0,
        "eyes_motion_p95": 9.0,
        "eyes_motion_max_over_mean": 1.7,
        "eyes_per_pair_motion": [4.0, 7.0],
        "blink_like_events": 2,
        "blink_detected": True,
        "estimated_blink_count": 2,
        "blink_confidence": 0.8,
        "eye_openness_series": [0.9, 0.8, 0.1, 0.9],
        "openness_threshold": 0.3,
        "blink_method": "dip_detection",
        "eye_openness_range": 0.8,
        "blink_dip_fraction": 0.25,
        "boundary_face_ratio_mean": 1.2,
        "boundary_face_ratio_std": 0.1,
        "boundary_face_ratio_series": [1.1, 1.3],
        "boundary_method": "edge_energy",
        "boundary_bg_ratio_mean": 0.8,
        "boundary_bg_ratio_std": 0.05,
        "boundary_bg_ratio_series": [0.75, 0.85],
        "boundary_face_over_bg_mean": 1.5,
        "boundary_face_minus_bg_mean": 0.4,
        "notes": [],
    }

    @pytest.fixture
    def mock_evidence(self):
        """Mock evidence data."""
        return {
            "manifest_path": "/fake/manifest.json",
            "frames_dir": "/fake/frames",
            "sampled_frame_count": 12,
            "sampled_frame_files": ["frame_0000.jpg", "frame_0075.jpg"],
            "roi_method": "haar_face_per_frame",
            "face_bbox": {"x": 100, "y": 100, "w": 200, "h": 200},
            "face_bbox_series": [{"x": 100, "y": 100, "w": 200, "h": 200}],
            "mouth_roi": {"x": 120, "y": 180, "w": 60, "h": 40},
            "eyes_roi": {"x": 120, "y": 120, "w": 60, "h": 40},
            "face_bbox_none_ratio": 0.0,
            "face_bbox_center_jitter_mean": 5.2,
            "face_bbox_size_jitter_mean": 3.1,
            "global_motion_mean": 12.5,
            "global_motion_min": 8.0,
            "global_motion_max": 18.0,
            "global_motion_p95": 16.0,
            "global_per_pair_motion": [10.0, 15.0],
            "mouth_motion_mean": 14.2,
            "mouth_motion_min": 10.0,
            "mouth_motion_max": 20.0,
            "mouth_motion_p95": 18.0,
            "mouth_motion_max_over_mean": 1.4,
            "mouth_per_pair_motion": [12.0, 16.0],
            "eyes_motion_mean": 5.8,
            "eyes_motion_min": 3.0,
            "eyes_motion_max": 10.0,
            "eyes_motion_p95": 9.0,
            "eyes_motion_max_over_mean": 1.7,
            "eyes_per_pair_motion": [4.0, 7.0],
            "blink_like_events": 2,
            "blink_detected": True,
            "estimated_blink_count": 2,
            "blink_confidence": 0.8,
            "eye_openness_series": [0.9, 0.8, 0.1, 0.9],
            "openness_threshold": 0.3,
            "blink_method": "dip_detection",
            "eye_openness_range": 0.8,
            "blink_dip_fraction": 0.25,
            "boundary_face_ratio_mean": 1.2,
            "boundary_face_ratio_std": 0.1,
            "boundary_face_ratio_series": [1.1, 1.3],
            "boundary_method": "edge_energy",
            "boundary_bg_ratio_mean": 0.8,
            "boundary_bg_ratio_std": 0.05,
            "boundary_bg_ratio_series": [0.75, 0.85],
            "boundary_face_over_bg_mean": 1.5,
            "boundary_face_minus_bg_mean": 0.4,
            "notes": [],
        }

    @pytest.fixture
    def mock_llm_input(self):
        """Mock LLM input data."""
        return {
            "video_id": "/fake/video.mp4",
            "sampling": {
                "sampled_frame_count": 12,
                "sampling_mode": "uniform",
                "fps": 30.0,
                "duration_seconds": 30.0,
            },
            "evidence": {
                "roi_method": "haar_face_per_frame",
                "global_motion_mean": 12.5,
                "mouth_motion_mean": 14.2,
                "eyes_motion_mean": 5.8,
                "blink_detected": True,
                "boundary_face_over_bg_mean": 1.5,
            },
            "keyframes": [
                {
                    "sample_pos": 0,
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "filename": "frame_0000.jpg",
                    "path": "/fake/frames/frame_0000.jpg",
                },
                {
                    "sample_pos": 6,
                    "frame_index": 375,
                    "timestamp_seconds": 12.5,
                    "filename": "frame_0375.jpg",
                    "path": "/fake/frames/frame_0375.jpg",
                },
            ],
        }

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response."""
        mock_response = MagicMock()
        mock_response.raw_text = "Label: MANIPULATED\nReason: Suspicious motion patterns detected."
        return mock_response

    @pytest.fixture
    def mock_decision(self):
        """Mock parsed decision."""
        return Decision(
            label="MANIPULATED",
            reason="Suspicious motion patterns detected.",
            raw_text="Label: MANIPULATED\nReason: Suspicious motion patterns detected.",
        )

    @patch("deepfake_detector.pipeline.extract_frames")
    @patch("deepfake_detector.pipeline.compute_basic_signals")
    @patch("deepfake_detector.pipeline.save_basic_signals")
    @patch("deepfake_detector.pipeline.build_llm_input")
    @patch("deepfake_detector.pipeline.save_llm_input")
    @patch("deepfake_detector.pipeline.build_prompt_text")
    @patch("deepfake_detector.pipeline.MockLLMClient")
    @patch("deepfake_detector.pipeline.parse_llm_output")
    def test_run_pipeline_success_mock_backend(
        self,
        mock_parse_llm_output,
        mock_mock_client_class,
        mock_build_prompt_text,
        mock_save_llm_input,
        mock_build_llm_input,
        mock_save_basic_signals,
        mock_compute_basic_signals,
        mock_extract_frames,
        mock_evidence,
        mock_llm_input,
        mock_llm_response,
        mock_decision,
    ):
        """Test successful pipeline run with mock backend."""
        # Setup mocks
        mock_extract_frames.return_value = self.mock_manifest_data
        mock_compute_basic_signals.return_value = self.mock_evidence_data
        mock_build_llm_input.return_value = mock_llm_input
        mock_build_prompt_text.return_value = "Mock prompt text"
        mock_client_instance = MagicMock()
        mock_client_instance.generate.return_value = mock_llm_response
        mock_mock_client_class.return_value = mock_client_instance
        mock_parse_llm_output.return_value = mock_decision

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the manifest file since extract_frames is mocked
            frames_dir = Path(temp_dir) / "frames"
            frames_dir.mkdir(parents=True)
            manifest_path = frames_dir / "frames_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(self.mock_manifest_data, f)
            # Create the evidence file since save_basic_signals is mocked
            evidence_path = frames_dir / "evidence_basic.json"
            with open(evidence_path, "w") as f:
                json.dump(self.mock_evidence_data, f)
            result = run_pipeline(
                video_path="/fake/video.mp4",
                out_dir=temp_dir,
                llm_backend="mock",
                num_frames=12,
                max_keyframes=8,
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert result["video_path"] == "/fake/video.mp4"
            assert result["out_dir"] == temp_dir
            assert "manifest_path" in result
            assert "evidence_path" in result
            assert "llm_input_path" in result
            assert "prompt_path" in result
            assert "llm_output_path" in result
            assert "decision_path" in result
            assert result["label"] == "MANIPULATED"
            assert result["reason"] == "Suspicious motion patterns detected."

            # Verify all paths exist
            for path_key in [
                "manifest_path",
                "evidence_path",
                "prompt_path",
                "llm_output_path",
                "decision_path",
            ]:
                assert Path(result[path_key]).exists()

            # Verify function calls
            mock_extract_frames.assert_called_once()
            mock_compute_basic_signals.assert_called_once()
            mock_save_basic_signals.assert_called_once()
            mock_build_llm_input.assert_called_once()
            mock_save_llm_input.assert_called_once()
            mock_build_prompt_text.assert_called_once()
            mock_mock_client_class.assert_called_once()
            mock_client_instance.generate.assert_called_once()
            mock_parse_llm_output.assert_called_once()

    @patch("deepfake_detector.pipeline.extract_frames")
    @patch("deepfake_detector.pipeline.compute_basic_signals")
    def test_run_pipeline_creates_output_directory(
        self, mock_compute_basic_signals, mock_extract_frames, mock_evidence
    ):
        """Test that pipeline creates output directory if it doesn't exist."""
        mock_extract_frames.return_value = self.mock_manifest_data
        mock_compute_basic_signals.return_value = mock_evidence

        with tempfile.TemporaryDirectory() as temp_base:
            out_dir = Path(temp_base) / "nested" / "output" / "dir"

            # Directory shouldn't exist yet
            assert not out_dir.exists()

            # Create the manifest file since extract_frames is mocked
            frames_dir = out_dir / "frames"
            frames_dir.mkdir(parents=True)
            manifest_path = frames_dir / "frames_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(self.mock_manifest_data, f)

            with (
                patch("deepfake_detector.pipeline.save_basic_signals"),
                patch("deepfake_detector.pipeline.build_llm_input"),
                patch("deepfake_detector.pipeline.save_llm_input"),
                patch("deepfake_detector.pipeline.build_prompt_text", return_value="mock prompt"),
                patch("deepfake_detector.pipeline.MockLLMClient") as mock_client_class,
                patch("deepfake_detector.pipeline.parse_llm_output") as mock_parse,
            ):

                mock_llm_response = MagicMock()
                mock_llm_response.raw_text = "mock response"
                mock_client_instance = MagicMock()
                mock_client_instance.generate.return_value = mock_llm_response
                mock_client_class.return_value = mock_client_instance
                mock_parse.return_value = Decision("REAL", "mock", "mock")

                run_pipeline(video_path="/fake/video.mp4", out_dir=str(out_dir))

            # Directory should now exist
            assert out_dir.exists()
            assert out_dir.is_dir()

    @patch("deepfake_detector.pipeline.extract_frames")
    @patch("deepfake_detector.pipeline.compute_basic_signals")
    @patch("deepfake_detector.pipeline.save_basic_signals")
    @patch("deepfake_detector.pipeline.build_llm_input")
    @patch("deepfake_detector.pipeline.save_llm_input")
    @patch("deepfake_detector.pipeline.build_prompt_text")
    @patch("deepfake_detector.llm.azure_client.AzureOpenAIClient")
    @patch("deepfake_detector.pipeline.parse_llm_output")
    def test_run_pipeline_azure_backend(
        self,
        mock_parse_llm_output,
        mock_azure_client_class,
        mock_build_prompt_text,
        mock_save_llm_input,
        mock_build_llm_input,
        mock_save_basic_signals,
        mock_compute_basic_signals,
        mock_extract_frames,
        mock_evidence,
        mock_llm_input,
        mock_llm_response,
        mock_decision,
    ):
        """Test pipeline run with Azure backend."""
        # Setup mocks
        mock_extract_frames.return_value = self.mock_manifest_data
        mock_compute_basic_signals.return_value = self.mock_evidence_data
        mock_build_llm_input.return_value = mock_llm_input
        mock_build_prompt_text.return_value = "Mock prompt text"
        mock_client_instance = MagicMock()
        mock_client_instance.generate.return_value = mock_llm_response
        mock_azure_client_class.return_value = mock_client_instance
        mock_parse_llm_output.return_value = mock_decision

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the manifest file since extract_frames is mocked
            frames_dir = Path(temp_dir) / "frames"
            frames_dir.mkdir(parents=True)
            manifest_path = frames_dir / "frames_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(self.mock_manifest_data, f)

            run_pipeline(
                video_path="/fake/video.mp4",
                out_dir=temp_dir,
                llm_backend="azure",
                num_frames=12,
                max_keyframes=8,
            )

            # Verify Azure client was used
            mock_azure_client_class.assert_called_once()
            mock_client_instance.generate.assert_called_once()

    @patch("deepfake_detector.pipeline.extract_frames")
    @patch("deepfake_detector.pipeline.compute_basic_signals")
    def test_run_pipeline_invalid_backend(self, mock_compute_basic_signals, mock_extract_frames):
        """Test pipeline with invalid LLM backend raises ValueError."""
        mock_extract_frames.return_value = self.mock_manifest_data
        mock_compute_basic_signals.return_value = self.mock_evidence_data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the manifest file since extract_frames is mocked
            frames_dir = Path(temp_dir) / "frames"
            frames_dir.mkdir(parents=True)
            manifest_path = frames_dir / "frames_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(self.mock_manifest_data, f)

            with pytest.raises(ValueError, match="Unsupported llm_backend: invalid"):
                run_pipeline(video_path="/fake/video.mp4", out_dir=temp_dir, llm_backend="invalid")

    @patch("deepfake_detector.pipeline.extract_frames")
    @patch("deepfake_detector.pipeline.compute_basic_signals")
    @patch("deepfake_detector.pipeline.save_basic_signals")
    @patch("deepfake_detector.pipeline.build_llm_input")
    @patch("deepfake_detector.pipeline.save_llm_input")
    @patch("deepfake_detector.pipeline.build_prompt_text")
    @patch("deepfake_detector.pipeline.MockLLMClient")
    @patch("deepfake_detector.pipeline.parse_llm_output")
    def test_run_pipeline_saves_artifacts_correctly(
        self,
        mock_parse_llm_output,
        mock_mock_client_class,
        mock_build_prompt_text,
        mock_save_llm_input,
        mock_build_llm_input,
        mock_save_basic_signals,
        mock_compute_basic_signals,
        mock_extract_frames,
        mock_evidence,
        mock_llm_input,
        mock_llm_response,
        mock_decision,
    ):
        """Test that all artifacts are saved with correct content."""
        # Setup mocks
        mock_extract_frames.return_value = self.mock_manifest_data
        mock_compute_basic_signals.return_value = self.mock_evidence_data
        mock_build_llm_input.return_value = mock_llm_input
        mock_build_prompt_text.return_value = "Test prompt content"
        mock_client_instance = MagicMock()
        mock_llm_response.raw_text = "Label: REAL\nReason: Natural motion detected."
        mock_client_instance.generate.return_value = mock_llm_response
        mock_mock_client_class.return_value = mock_client_instance
        mock_parse_llm_output.return_value = Decision(
            label="REAL",
            reason="Natural motion detected.",
            raw_text="Label: REAL\nReason: Natural motion detected.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the manifest file since extract_frames is mocked
            frames_dir = Path(temp_dir) / "frames"
            frames_dir.mkdir(parents=True)
            manifest_path = frames_dir / "frames_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(self.mock_manifest_data, f)

            result = run_pipeline(
                video_path="/fake/video.mp4", out_dir=temp_dir, llm_backend="mock"
            )

            # Check manifest was saved
            manifest_path = Path(result["manifest_path"])
            assert manifest_path.exists()
            with open(manifest_path, "r") as f:
                saved_manifest = json.load(f)
            assert saved_manifest == self.mock_manifest_data

            # Check prompt was saved
            prompt_path = Path(result["prompt_path"])
            assert prompt_path.exists()
            assert prompt_path.read_text() == "Test prompt content"

            # Check LLM output was saved
            llm_output_path = Path(result["llm_output_path"])
            assert llm_output_path.exists()
            assert llm_output_path.read_text() == "Label: REAL\nReason: Natural motion detected."

            # Check decision was saved
            decision_path = Path(result["decision_path"])
            assert decision_path.exists()
            with open(decision_path, "r") as f:
                saved_decision = json.load(f)
            assert saved_decision["label"] == "REAL"
            assert saved_decision["reason"] == "Natural motion detected."
            assert "raw_text" in saved_decision

    @patch("deepfake_detector.pipeline.extract_frames")
    @patch("deepfake_detector.pipeline.compute_basic_signals")
    @patch("deepfake_detector.pipeline.save_basic_signals")
    @patch("deepfake_detector.pipeline.build_llm_input")
    @patch("deepfake_detector.pipeline.save_llm_input")
    @patch("deepfake_detector.pipeline.build_prompt_text")
    @patch("deepfake_detector.pipeline.MockLLMClient")
    @patch("deepfake_detector.pipeline.parse_llm_output")
    def test_run_pipeline_with_custom_parameters(
        self,
        mock_parse_llm_output,
        mock_mock_client_class,
        mock_build_prompt_text,
        mock_save_llm_input,
        mock_build_llm_input,
        mock_save_basic_signals,
        mock_compute_basic_signals,
        mock_extract_frames,
        mock_evidence,
        mock_llm_input,
        mock_llm_response,
        mock_decision,
    ):
        """Test pipeline with custom num_frames and max_keyframes."""
        # Setup mocks
        mock_extract_frames.return_value = self.mock_manifest_data
        mock_compute_basic_signals.return_value = self.mock_evidence_data
        mock_build_llm_input.return_value = mock_llm_input
        mock_build_prompt_text.return_value = "Mock prompt"
        mock_client_instance = MagicMock()
        mock_client_instance.generate.return_value = mock_llm_response
        mock_mock_client_class.return_value = mock_client_instance
        mock_parse_llm_output.return_value = mock_decision

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the manifest file since extract_frames is mocked
            frames_dir = Path(temp_dir) / "frames"
            frames_dir.mkdir(parents=True)
            manifest_path = frames_dir / "frames_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(self.mock_manifest_data, f)

            run_pipeline(
                video_path="/fake/video.mp4",
                out_dir=temp_dir,
                llm_backend="mock",
                num_frames=20,
                max_keyframes=4,
            )

            # Verify extract_frames was called with correct num_frames
            mock_extract_frames.assert_called_once()
            call_args = mock_extract_frames.call_args
            assert call_args[1]["num_frames"] == 20  # kwargs

            # Verify build_llm_input was called with correct max_keyframes
            mock_build_llm_input.assert_called_once()
            call_args = mock_build_llm_input.call_args
            assert call_args[1]["max_keyframes"] == 4  # kwargs
