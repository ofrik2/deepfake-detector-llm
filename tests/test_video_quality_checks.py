import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.deepfake_detector.video.quality_checks import (
    _blur_score_laplacian,
    _mean_brightness,
    run_quality_checks,
)


class TestMeanBrightness:
    def test_grayscale_image(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        assert _mean_brightness(img) == 128.0

    def test_color_image(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        assert _mean_brightness(img) == 100.0

    def test_varied_brightness(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:5, :] = 255
        expected = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        assert _mean_brightness(img) == pytest.approx(expected)


class TestBlurScoreLaplacian:
    def test_sharp_image(self):
        # Checkerboard pattern should be sharp
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        score = _blur_score_laplacian(img)
        assert score > 0

    def test_blurry_image(self):
        # Uniform image should have low variance
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        score = _blur_score_laplacian(img)
        assert score == 0.0

    def test_color_image(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        score = _blur_score_laplacian(img)
        assert score == 0.0


class TestRunQualityChecks:
    @patch('src.deepfake_detector.video.quality_checks.open_video')
    @patch('src.deepfake_detector.video.quality_checks.read_video_meta')
    @patch('src.deepfake_detector.video.quality_checks.read_frame_at')
    def test_successful_checks(self, mock_read_frame, mock_read_meta, mock_open_video):
        # Mock video meta
        mock_meta = MagicMock()
        mock_meta.fps = 30.0
        mock_meta.frame_count = 10
        mock_meta.width = 640
        mock_meta.height = 480
        mock_meta.duration_seconds = 10 / 30
        mock_read_meta.return_value = mock_meta

        # Mock cap
        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        # Mock frames: bright, sharp frames
        mock_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        # Add some pattern for sharpness
        mock_frame[::10, :] = 0
        mock_read_frame.return_value = (True, mock_frame)

        result = run_quality_checks('dummy.mp4')

        assert result['ok'] is True
        assert len(result['errors']) == 0
        assert 'brightness_mean' in result['stats']
        assert 'blur_mean' in result['stats']
        assert result['stats']['sampled_frames'] == [0, 2, 4, 7, 9]  # For sample_frames=5

    @patch('src.deepfake_detector.video.quality_checks.open_video')
    @patch('src.deepfake_detector.video.quality_checks.read_video_meta')
    @patch('src.deepfake_detector.video.quality_checks.read_frame_at')
    def test_too_few_frames_error(self, mock_read_frame, mock_read_meta, mock_open_video):
        mock_meta = MagicMock()
        mock_meta.frame_count = 5  # Less than min_frame_count=10
        mock_meta.width = 640
        mock_meta.height = 480
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        result = run_quality_checks('dummy.mp4', min_frame_count=10)

        assert result['ok'] is False
        assert len(result['errors']) == 1
        assert 'Too few frames' in result['errors'][0]

    @patch('src.deepfake_detector.video.quality_checks.open_video')
    @patch('src.deepfake_detector.video.quality_checks.read_video_meta')
    @patch('src.deepfake_detector.video.quality_checks.read_frame_at')
    def test_low_resolution_warning(self, mock_read_frame, mock_read_meta, mock_open_video):
        mock_meta = MagicMock()
        mock_meta.frame_count = 20
        mock_meta.width = 100  # Less than min_resolution
        mock_meta.height = 100
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        result = run_quality_checks('dummy.mp4', min_resolution=(200, 200))

        assert result['ok'] is True
        assert len(result['warnings']) >= 1
        assert 'Low resolution' in ' '.join(result['warnings'])

    @patch('src.deepfake_detector.video.quality_checks.open_video')
    @patch('src.deepfake_detector.video.quality_checks.read_video_meta')
    @patch('src.deepfake_detector.video.quality_checks.read_frame_at')
    def test_dark_video_warning(self, mock_read_frame, mock_read_meta, mock_open_video):
        mock_meta = MagicMock()
        mock_meta.frame_count = 10
        mock_meta.width = 640
        mock_meta.height = 480
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        # Dark frame
        mock_frame = np.full((100, 100, 3), 10, dtype=np.uint8)  # Below brightness_range[0]=20
        mock_read_frame.return_value = (True, mock_frame)

        result = run_quality_checks('dummy.mp4', sample_frames=1)

        assert 'dark' in ' '.join(result['warnings']).lower()

    @patch('src.deepfake_detector.video.quality_checks.open_video')
    @patch('src.deepfake_detector.video.quality_checks.read_video_meta')
    @patch('src.deepfake_detector.video.quality_checks.read_frame_at')
    def test_bright_video_warning(self, mock_read_frame, mock_read_meta, mock_open_video):
        mock_meta = MagicMock()
        mock_meta.frame_count = 10
        mock_meta.width = 640
        mock_meta.height = 480
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        # Bright frame
        mock_frame = np.full((100, 100, 3), 250, dtype=np.uint8)  # Above brightness_range[1]=235
        mock_read_frame.return_value = (True, mock_frame)

        result = run_quality_checks('dummy.mp4', sample_frames=1)

        assert 'bright' in ' '.join(result['warnings']).lower()

    @patch('src.deepfake_detector.video.quality_checks.open_video')
    @patch('src.deepfake_detector.video.quality_checks.read_video_meta')
    @patch('src.deepfake_detector.video.quality_checks.read_frame_at')
    def test_blurry_video_warning(self, mock_read_frame, mock_read_meta, mock_open_video):
        mock_meta = MagicMock()
        mock_meta.frame_count = 10
        mock_meta.width = 640
        mock_meta.height = 480
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        # Blurry frame (uniform color)
        mock_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        result = run_quality_checks('dummy.mp4', sample_frames=1, min_blur_score=50.0)

        assert 'blurry' in ' '.join(result['warnings']).lower()

    @patch('src.deepfake_detector.video.quality_checks.open_video')
    @patch('src.deepfake_detector.video.quality_checks.read_video_meta')
    @patch('src.deepfake_detector.video.quality_checks.read_frame_at')
    def test_unreadable_frames(self, mock_read_frame, mock_read_meta, mock_open_video):
        mock_meta = MagicMock()
        mock_meta.frame_count = 10
        mock_meta.width = 640
        mock_meta.height = 480
        mock_read_meta.return_value = mock_meta

        mock_cap = MagicMock()
        mock_open_video.return_value = mock_cap

        # Some frames unreadable
        mock_read_frame.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8)),
                                       (False, None),
                                       (True, np.zeros((100, 100, 3), dtype=np.uint8))]

        result = run_quality_checks('dummy.mp4', sample_frames=3)

        assert result['stats']['unreadable_samples'] == 1

    def test_no_sample_frames(self):
        with patch('src.deepfake_detector.video.quality_checks.open_video') as mock_open:
            with patch('src.deepfake_detector.video.quality_checks.read_video_meta') as mock_meta:
                mock_meta.return_value = MagicMock(frame_count=10, width=640, height=480)
                mock_cap = MagicMock()
                mock_open.return_value = mock_cap

                result = run_quality_checks('dummy.mp4', sample_frames=0)

                assert 'brightness_mean' not in result['stats']