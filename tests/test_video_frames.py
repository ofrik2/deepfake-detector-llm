import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.deepfake_detector.video.frames import (
    _ensure_dir,
    _every_n_indices,
    _resize_keep_aspect,
    _uniform_indices,
    extract_frames,
    FrameRecord,
    FramesManifest,
)


class TestEnsureDir:
    def test_creates_directory_if_not_exists(self, tmp_path):
        dir_path = tmp_path / "new_dir"
        _ensure_dir(str(dir_path))
        assert dir_path.exists()
        assert dir_path.is_dir()

    def test_does_nothing_if_exists(self, tmp_path):
        dir_path = tmp_path / "existing_dir"
        dir_path.mkdir()
        _ensure_dir(str(dir_path))
        assert dir_path.exists()


class TestResizeKeepAspect:
    def test_no_resize_if_smaller(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _resize_keep_aspect(frame, 200)
        assert result.shape == (100, 100, 3)
        assert np.array_equal(result, frame)

    def test_resize_down(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        result = _resize_keep_aspect(frame, 200)
        assert result.shape == (200, 200, 3)

    def test_resize_max_zero(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _resize_keep_aspect(frame, 0)
        assert result.shape == (100, 100, 3)


class TestUniformIndices:
    def test_empty_for_zero_frames(self):
        assert _uniform_indices(0, 5) == []

    def test_empty_for_zero_num(self):
        assert _uniform_indices(100, 0) == []

    def test_single_frame(self):
        assert _uniform_indices(1, 1) == [0]

    def test_more_num_than_frames(self):
        assert _uniform_indices(3, 5) == [0, 1, 2]

    def test_uniform_distribution(self):
        indices = _uniform_indices(100, 5)
        expected = [0, 25, 50, 74, 99]  # rounded evenly
        assert indices == expected

    def test_two_frames(self):
        assert _uniform_indices(10, 2) == [0, 9]


class TestEveryNIndices:
    def test_empty_cases(self):
        assert _every_n_indices(0, 5, 10) == []
        assert _every_n_indices(10, 0, 10) == []

    def test_basic(self):
        assert _every_n_indices(20, 5, 100) == [0, 5, 10, 15]

    def test_capped_by_max_frames(self):
        assert _every_n_indices(100, 1, 3) == [0, 1, 2]


class TestExtractFrames:
    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    @patch('src.deepfake_detector.video.frames.read_frame_at')
    @patch('src.deepfake_detector.video.frames.cv2.imwrite')
    def test_extract_uniform_mode(self, mock_imwrite, mock_read_frame, mock_read_meta, mock_open_video, tmp_path):
        # Mock video meta
        mock_meta = MagicMock()
        mock_meta.frame_count = 10
        mock_meta.fps = 30.0
        mock_meta.width = 640
        mock_meta.height = 480
        mock_meta.duration_seconds = 10 / 30
        mock_meta.path = 'dummy.mp4'
        mock_read_meta.return_value = mock_meta

        # Mock cap
        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        # Mock frames
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)
        mock_imwrite.return_value = True

        out_dir = tmp_path / "frames"
        result = extract_frames('dummy.mp4', str(out_dir), mode='uniform', num_frames=3)

        assert 'frames' in result
        assert len(result['frames']) == 3
        assert result['sampling_mode'] == 'uniform'
        assert result['num_frames_requested'] == 3

        # Check calls
        mock_open_video.assert_called_once_with('dummy.mp4')
        mock_read_meta.assert_called_once()
        assert mock_read_frame.call_count == 3
        assert mock_imwrite.call_count == 3
        mock_cap.release.assert_called_once()

    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    @patch('src.deepfake_detector.video.frames.read_frame_at')
    @patch('src.deepfake_detector.video.frames.cv2.imwrite')
    def test_extract_every_n_mode(self, mock_imwrite, mock_read_frame, mock_read_meta, mock_open_video, tmp_path):
        mock_meta = MagicMock()
        mock_meta.frame_count = 20
        mock_meta.fps = 30.0
        mock_meta.width = 640
        mock_meta.height = 480
        mock_meta.duration_seconds = 20 / 30
        mock_meta.path = 'dummy.mp4'
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)
        mock_imwrite.return_value = True

        out_dir = tmp_path / "frames"
        result = extract_frames('dummy.mp4', str(out_dir), mode='every_n', every_n=5, max_frames=10)

        assert len(result['frames']) == 4  # 0,5,10,15
        assert result['sampling_mode'] == 'every_n'

    @patch('src.deepfake_detector.video.frames.read_video_meta')
    @patch('src.deepfake_detector.video.frames.open_video')
    def test_invalid_mode(self, mock_open_video, mock_read_meta, tmp_path):
        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap
        mock_meta = MagicMock()
        mock_meta.frame_count = 10
        mock_read_meta.return_value = mock_meta

        with pytest.raises(ValueError, match="Unknown sampling mode"):
            extract_frames('dummy.mp4', str(tmp_path), mode='invalid')

    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    def test_no_frames_selected(self, mock_read_meta, mock_open_video, tmp_path):
        mock_meta = MagicMock()
        mock_meta.frame_count = 10
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        with pytest.raises(ValueError, match="No frames selected"):
            extract_frames('dummy.mp4', str(tmp_path), num_frames=0)

    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    @patch('src.deepfake_detector.video.frames.read_frame_at')
    @patch('src.deepfake_detector.video.frames.cv2.imwrite')
    def test_failed_frame_read(self, mock_imwrite, mock_read_frame, mock_read_meta, mock_open_video, tmp_path):
        mock_meta = MagicMock()
        mock_meta.configure_mock(frame_count=5, fps=30.0, width=640, height=480, duration_seconds=5/30, path="dummy.mp4")
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        # First read succeeds, second fails
        mock_read_frame.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8)), (False, None)]
        mock_imwrite.return_value = True

        result = extract_frames('dummy.mp4', str(tmp_path), num_frames=2)

        # Should skip the failed frame and continue
        assert len(result['frames']) == 1  # Only first frame

    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    @patch('src.deepfake_detector.video.frames.read_frame_at')
    @patch('src.deepfake_detector.video.frames.cv2.imwrite')
    def test_failed_write(self, mock_imwrite, mock_read_frame, mock_read_meta, mock_open_video, tmp_path):
        mock_meta = MagicMock()
        mock_meta.configure_mock(frame_count=2, fps=30.0, width=640, height=480, duration_seconds=2/30, path="dummy.mp4")
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)
        mock_imwrite.return_value = False  # Write fails

        with pytest.raises(OSError, match="Failed to write frame"):
            extract_frames('dummy.mp4', str(tmp_path), num_frames=2)

    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    def test_no_frames_after_filtering(self, mock_read_meta, mock_open_video, tmp_path):
        mock_meta = MagicMock()
        mock_meta.frame_count = 2
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        with patch('src.deepfake_detector.video.frames.read_frame_at') as mock_read:
            mock_read.return_value = (False, None)  # All reads fail

            with pytest.raises(ValueError, match="Failed to extract any frames"):
                extract_frames('dummy.mp4', str(tmp_path), num_frames=2)

    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    @patch('src.deepfake_detector.video.frames.read_frame_at')
    @patch('src.deepfake_detector.video.frames.cv2.imwrite')
    @patch('src.deepfake_detector.video.frames.json.dump')
    def test_write_manifest(self, mock_json_dump, mock_imwrite, mock_read_frame, mock_read_meta, mock_open_video, tmp_path):
        mock_meta = MagicMock()
        mock_meta.frame_count = 1
        mock_meta.fps = 30.0
        mock_meta.width = 640
        mock_meta.height = 480
        mock_meta.duration_seconds = 1/30
        mock_meta.path = 'dummy.mp4'
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)
        mock_imwrite.return_value = True

        out_dir = tmp_path / "frames"
        extract_frames('dummy.mp4', str(out_dir), num_frames=1, write_manifest=True)

        # Check manifest was written
        manifest_path = out_dir / "frames_manifest.json"
        assert manifest_path.exists()
        mock_json_dump.assert_called_once()

    @patch('src.deepfake_detector.video.frames.open_video')
    @patch('src.deepfake_detector.video.frames.read_video_meta')
    @patch('src.deepfake_detector.video.frames.read_frame_at')
    @patch('src.deepfake_detector.video.frames.cv2.imwrite')
    def test_resize_frames(self, mock_imwrite, mock_read_frame, mock_read_meta, mock_open_video, tmp_path):
        mock_meta = MagicMock()
        mock_meta.frame_count = 1
        mock_meta.fps = 30.0
        mock_meta.width = 1000
        mock_meta.height = 1000
        mock_meta.duration_seconds = 1/30
        mock_meta.path = 'dummy.mp4'
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        mock_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)
        mock_imwrite.return_value = True

        out_dir = tmp_path / "frames"
        extract_frames('dummy.mp4', str(out_dir), num_frames=1, resize_max=500)

        # Check that resize was called
        args, kwargs = mock_imwrite.call_args
        written_frame = args[1]
        assert written_frame.shape == (500, 500, 3)
