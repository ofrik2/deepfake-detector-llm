from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.deepfake_detector.video.reader import (
    VideoMeta,
    open_video,
    read_frame_at,
    read_video_meta,
)


class TestOpenVideo:
    @patch("src.deepfake_detector.video.reader.cv2.VideoCapture")
    @patch("src.deepfake_detector.video.reader.os.path.exists", return_value=True)
    def test_successful_open(self, mock_exists, mock_cv2_cap):
        mock_cap = MagicMock()
        mock_cv2_cap.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        result = open_video("test.mp4")

        mock_cv2_cap.assert_called_once_with("test.mp4")
        assert result == mock_cap

    @patch("src.deepfake_detector.video.reader.cv2.VideoCapture")
    def test_file_not_found(self, mock_cv2_cap):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Video not found"):
                open_video("nonexistent.mp4")

    @patch("src.deepfake_detector.video.reader.cv2.VideoCapture")
    @patch("src.deepfake_detector.video.reader.os.path.exists", return_value=True)
    def test_failed_to_open(self, mock_exists, mock_cv2_cap):
        mock_cap = MagicMock()
        mock_cv2_cap.return_value = mock_cap
        mock_cap.isOpened.return_value = False

        with pytest.raises(OSError, match="Failed to open video"):
            open_video("test.mp4")


class TestReadVideoMeta:
    def test_with_provided_cap(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)

        result = read_video_meta("test.mp4", cap=mock_cap)

        assert isinstance(result, VideoMeta)
        assert result.path == "test.mp4"
        assert result.fps == 30.0
        assert result.frame_count == 900
        assert result.width == 1920
        assert result.height == 1080
        assert result.duration_seconds == 30.0

        mock_cap.release.assert_not_called()  # Since cap was provided

    def test_without_cap(self):
        with patch("src.deepfake_detector.video.reader.open_video") as mock_open:
            mock_cap = MagicMock()
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_COUNT: 250,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
            }.get(prop, 0)
            mock_open.return_value = mock_cap

            result = read_video_meta("test.mp4")

            assert result.fps == 25.0
            assert result.duration_seconds == 10.0
            mock_cap.release.assert_called_once()

    def test_zero_fps(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 0.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)

        result = read_video_meta("test.mp4", cap=mock_cap)

        assert result.fps == 0.0
        assert result.duration_seconds == 0.0

    def test_zero_frame_count(self):
        mock_cap = MagicMock()
        mock_cap.get.return_value = 0

        result = read_video_meta("test.mp4", cap=mock_cap)

        assert result.frame_count == 0
        assert result.duration_seconds == 0.0


class TestReadFrameAt:
    def test_successful_read(self):
        mock_cap = MagicMock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = None

        ok, frame = read_frame_at(mock_cap, 10)

        assert ok is True
        assert frame is mock_frame
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 10)
        mock_cap.read.assert_called_once()

    def test_failed_read(self):
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)

        ok, frame = read_frame_at(mock_cap, 5)

        assert ok is False
        assert frame is None

    def test_read_none_frame(self):
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, None)

        ok, frame = read_frame_at(mock_cap, 1)

        assert ok is True
        assert frame is None
