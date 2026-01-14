"""
test_cli.py

Unit tests for the CLI functionality.
Tests argument parsing and main function execution.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from deepfake_detector.cli import main


class TestCLI:
    """Test the CLI functionality."""

    @patch("deepfake_detector.cli.get_detector")
    @patch(
        "sys.argv",
        [
            "deepfake-detector-llm",
            "detect",
            "--video",
            "assets/videos/sample.mp4",
            "--out",
            "runs/test_run",
        ],
    )
    def test_main_detect_command_basic(self, mock_get_detector):
        """Test basic detect command execution."""
        mock_detector = MagicMock()
        mock_get_detector.return_value = lambda: mock_detector
        mock_detector.detect.return_value = MagicMock(
            label="REAL", rationale="Natural motion detected.", evidence_used=[], metadata={}
        )

        # Capture stdout
        with patch("builtins.print") as mock_print:
            main()

        # Verify detect was called
        mock_detector.detect.assert_called_once()

        # Verify output was printed
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        output_dict = json.loads(printed_output)
        assert output_dict["label"] == "REAL"

    @patch("deepfake_detector.cli.get_detector")
    @patch(
        "sys.argv",
        [
            "deepfake-detector-llm",
            "detect",
            "--video",
            "vid.mp4",
            "--out",
            "out_dir",
            "--llm",
            "azure",
            "--num-frames",
            "24",
            "--max-keyframes",
            "6",
        ],
    )
    def test_main_detect_command_with_options(self, mock_get_detector):
        """Test detect command with all optional arguments."""
        mock_detector = MagicMock()
        mock_get_detector.return_value = lambda: mock_detector
        mock_detector.detect.return_value = MagicMock(
            label="MANIPULATED", rationale="Suspicious", evidence_used=[], metadata={}
        )

        with patch("builtins.print"):
            main()

        # Verify detect was called with all specified arguments
        mock_detector.detect.assert_called_once_with(
            video_path="vid.mp4",
            out_dir="out_dir",
            config={"llm_backend": "azure", "num_frames": 24, "max_keyframes": 6},
        )

    @patch("sys.argv", ["deepfake-detector-llm"])
    def test_main_no_command(self):
        """Test that main fails when no command is provided."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch("sys.argv", ["deepfake-detector-llm", "invalid_command"])
    def test_main_invalid_command(self):
        """Test that main fails with invalid command."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch("sys.argv", ["deepfake-detector-llm", "detect"])
    def test_main_detect_missing_required_args(self):
        """Test that detect command fails when required args are missing."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch("sys.argv", ["deepfake-detector-llm", "detect", "--video", "vid.mp4"])
    def test_main_detect_missing_out_dir(self):
        """Test that detect command fails when out dir is missing."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch(
        "sys.argv",
        [
            "deepfake-detector-llm",
            "detect",
            "--video",
            "vid.mp4",
            "--out",
            "out_dir",
            "--llm",
            "invalid",
        ],
    )
    def test_main_detect_invalid_llm_backend(self):
        """Test that invalid LLM backend causes argparse to exit."""
        with pytest.raises(SystemExit):  # argparse exits on invalid choice
            main()

    @patch("deepfake_detector.cli.get_detector")
    @patch(
        "sys.argv", ["deepfake-detector-llm", "detect", "--video", "vid.mp4", "--out", "out_dir"]
    )
    def test_main_detect_pipeline_exception(self, mock_get_detector):
        """Test handling of exceptions during pipeline execution."""
        mock_detector = MagicMock()
        mock_get_detector.return_value = lambda: mock_detector
        mock_detector.detect.side_effect = FileNotFoundError("Video file not found")

        with patch("builtins.print") as mock_print:
            with pytest.raises(FileNotFoundError):
                main()

        # Verify error was not printed (exception propagates)
        mock_print.assert_not_called()

    @patch("deepfake_detector.cli.get_detector")
    @patch(
        "sys.argv", ["deepfake-detector-llm", "detect", "--video", "vid.mp4", "--out", "out_dir"]
    )
    def test_main_detect_output_formatting(self, mock_get_detector):
        """Test that output is properly formatted JSON."""
        mock_detector = MagicMock()
        mock_get_detector.return_value = lambda: mock_detector
        mock_detector.detect.return_value = MagicMock(
            label="UNCERTAIN", rationale="Insufficient evidence", evidence_used=[], metadata={}
        )

        with patch("builtins.print") as mock_print:
            main()

        # Verify output is valid JSON
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        parsed_output = json.loads(printed_output)
        assert isinstance(parsed_output, dict)
        assert parsed_output["label"] == "UNCERTAIN"
        assert parsed_output["rationale"] == "Insufficient evidence"

    def test_cli_help_output(self):
        """Test that help can be displayed without error."""
        with patch("sys.argv", ["deepfake-detector-llm", "--help"]):
            with pytest.raises(SystemExit):  # --help causes SystemExit
                main()

    def test_cli_detect_help_output(self):
        """Test that detect subcommand help can be displayed."""
        with patch("sys.argv", ["deepfake-detector-llm", "detect", "--help"]):
            with pytest.raises(SystemExit):  # --help causes SystemExit
                main()
