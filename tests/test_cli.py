"""
test_cli.py

Unit tests for the CLI functionality.
Tests argument parsing and main function execution.
"""

import json
from unittest.mock import patch

import pytest

from deepfake_detector.cli import main


class TestCLI:
    """Test the CLI functionality."""

    @patch('deepfake_detector.cli.run_pipeline')
    @patch('sys.argv', ['deepfake-detector-llm', 'detect', '--video', '/path/to/video.mp4', '--out', '/output/dir'])
    def test_main_detect_command_basic(self, mock_run_pipeline):
        """Test basic detect command execution."""
        mock_run_pipeline.return_value = {
            "video_path": "/path/to/video.mp4",
            "out_dir": "/output/dir",
            "label": "REAL",
            "reason": "Natural motion detected."
        }

        # Capture stdout
        with patch('builtins.print') as mock_print:
            main()

        # Verify run_pipeline was called with correct arguments
        mock_run_pipeline.assert_called_once_with(
            video_path="/path/to/video.mp4",
            out_dir="/output/dir",
            llm_backend="mock",
            num_frames=12,
            max_keyframes=8
        )

        # Verify output was printed
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        output_dict = json.loads(printed_output)
        assert output_dict["label"] == "REAL"

    @patch('deepfake_detector.cli.run_pipeline')
    @patch('sys.argv', ['deepfake-detector-llm', 'detect', '--video', '/video.mp4', '--out', '/out', '--llm', 'azure', '--num-frames', '24', '--max-keyframes', '6'])
    def test_main_detect_command_with_options(self, mock_run_pipeline):
        """Test detect command with all optional arguments."""
        mock_run_pipeline.return_value = {"label": "MANIPULATED"}

        with patch('builtins.print'):
            main()

        # Verify run_pipeline was called with all specified arguments
        mock_run_pipeline.assert_called_once_with(
            video_path="/video.mp4",
            out_dir="/out",
            llm_backend="azure",
            num_frames=24,
            max_keyframes=6
        )

    @patch('sys.argv', ['deepfake-detector-llm'])
    def test_main_no_command(self):
        """Test that main fails when no command is provided."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch('sys.argv', ['deepfake-detector-llm', 'invalid_command'])
    def test_main_invalid_command(self):
        """Test that main fails with invalid command."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch('sys.argv', ['deepfake-detector-llm', 'detect'])
    def test_main_detect_missing_required_args(self):
        """Test that detect command fails when required args are missing."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch('sys.argv', ['deepfake-detector-llm', 'detect', '--video', '/video.mp4'])
    def test_main_detect_missing_out_dir(self):
        """Test that detect command fails when out dir is missing."""
        with pytest.raises(SystemExit):  # argparse exits on error
            main()

    @patch('sys.argv', ['deepfake-detector-llm', 'detect', '--video', '/video.mp4', '--out', '/out', '--llm', 'invalid'])
    def test_main_detect_invalid_llm_backend(self):
        """Test that invalid LLM backend causes argparse to exit."""
        with pytest.raises(SystemExit):  # argparse exits on invalid choice
            main()

    @patch('deepfake_detector.cli.run_pipeline')
    @patch('sys.argv', ['deepfake-detector-llm', 'detect', '--video', '/video.mp4', '--out', '/out'])
    def test_main_detect_pipeline_exception(self, mock_run_pipeline):
        """Test handling of exceptions during pipeline execution."""
        mock_run_pipeline.side_effect = FileNotFoundError("Video file not found")

        with patch('builtins.print') as mock_print:
            with pytest.raises(FileNotFoundError):
                main()

        # Verify error was not printed (exception propagates)
        mock_print.assert_not_called()

    @patch('deepfake_detector.cli.run_pipeline')
    @patch('sys.argv', ['deepfake-detector-llm', 'detect', '--video', '/video.mp4', '--out', '/out'])
    def test_main_detect_output_formatting(self, mock_run_pipeline):
        """Test that output is properly formatted JSON."""
        mock_run_pipeline.return_value = {
            "video_path": "/video.mp4",
            "out_dir": "/out",
            "label": "UNCERTAIN",
            "reason": "Insufficient evidence",
            "manifest_path": "/out/frames/frames_manifest.json",
            "evidence_path": "/out/frames/evidence_basic.json"
        }

        with patch('builtins.print') as mock_print:
            main()

        # Verify output is valid JSON
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        parsed_output = json.loads(printed_output)
        assert isinstance(parsed_output, dict)
        assert parsed_output["label"] == "UNCERTAIN"
        assert parsed_output["reason"] == "Insufficient evidence"

    def test_cli_help_output(self):
        """Test that help can be displayed without error."""
        with patch('sys.argv', ['deepfake-detector-llm', '--help']):
            with pytest.raises(SystemExit):  # --help causes SystemExit
                main()

    def test_cli_detect_help_output(self):
        """Test that detect subcommand help can be displayed."""
        with patch('sys.argv', ['deepfake-detector-llm', 'detect', '--help']):
            with pytest.raises(SystemExit):  # --help causes SystemExit
                main()