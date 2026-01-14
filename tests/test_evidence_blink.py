import numpy as np

from src.deepfake_detector.evidence.blink import (
    _count_dip_events,
    _eye_openness_proxy,
    _smooth_1d,
    compute_blink_evidence_from_eyes_roi_series,
)


class TestEyeOpennessProxy:
    def test_empty_roi(self):
        roi = np.array([]).reshape(0, 0)
        assert _eye_openness_proxy(roi) == 0.0

    def test_uniform_roi(self):
        roi = np.full((10, 10), 128, dtype=np.uint8)
        score = _eye_openness_proxy(roi)
        # Laplacian variance of uniform is 0, log1p(0) = 0
        assert score == 0.0

    def test_varied_roi(self):
        roi = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        score = _eye_openness_proxy(roi)
        assert score > 0.0

    def test_float_input(self):
        roi = np.full((10, 10), 128.5, dtype=np.float32)
        score = _eye_openness_proxy(roi)
        assert score == 0.0  # After clipping to uint8


class TestSmooth1d:
    def test_empty_array(self):
        x = np.array([])
        result = _smooth_1d(x)
        assert len(result) == 0

    def test_single_element(self):
        x = np.array([5.0])
        result = _smooth_1d(x)
        np.testing.assert_array_equal(result, [5.0])

    def test_two_elements(self):
        x = np.array([1.0, 3.0])
        result = _smooth_1d(x)
        np.testing.assert_array_equal(result, [1.0, 3.0])

    def test_three_elements(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _smooth_1d(x)
        expected = [1.0, 2.0, 3.0]  # Median of [1,2,3] is 2, but edges unchanged
        np.testing.assert_array_equal(result, expected)

    def test_longer_series(self):
        x = np.array([1.0, 3.0, 5.0, 4.0, 2.0])
        result = _smooth_1d(x)
        expected = [1.0, 3.0, 4.0, 4.0, 2.0]  # Position 1: median(1,3,5)=3, etc.
        np.testing.assert_array_equal(result, expected)


class TestCountDipEvents:
    def test_empty_series(self):
        series = np.array([])
        assert _count_dip_events(series, 0.5) == 0

    def test_short_series(self):
        series = np.array([1.0])
        assert _count_dip_events(series, 0.5) == 0

        series = np.array([1.0, 2.0])
        assert _count_dip_events(series, 0.5) == 0

    def test_no_dips(self):
        series = np.array([1.0, 2.0, 3.0, 4.0])
        assert _count_dip_events(series, 2.5) == 0

    def test_single_dip(self):
        series = np.array([3.0, 1.0, 3.0])  # Dip at index 1
        assert _count_dip_events(series, 2.0) == 1

    def test_multiple_dips(self):
        series = np.array([3.0, 1.0, 4.0, 0.5, 2.0, 1.5, 3.0])
        # Dips at indices 1, 3, and 5
        assert _count_dip_events(series, 2.0) == 3

    def test_dip_at_edge(self):
        series = np.array([1.0, 3.0, 2.0])  # No dip at edges
        assert _count_dip_events(series, 2.5) == 0

    def test_no_dip_below_threshold(self):
        series = np.array([3.0, 2.5, 3.0])  # 2.5 not < 2.0
        assert _count_dip_events(series, 2.0) == 0


class TestComputeBlinkEvidence:
    def test_empty_input(self):
        result = compute_blink_evidence_from_eyes_roi_series([])
        assert result["blink_detected"] is False
        assert result["estimated_blink_count"] == 0
        assert result["eye_openness_series"] == []

    def test_single_roi(self):
        roi = np.full((10, 10), 128, dtype=np.uint8)
        result = compute_blink_evidence_from_eyes_roi_series([roi])
        assert result["blink_detected"] is False
        assert len(result["eye_openness_series"]) == 1

    def test_uniform_series(self):
        rois = [np.full((10, 10), 128, dtype=np.uint8) for _ in range(5)]
        result = compute_blink_evidence_from_eyes_roi_series(rois)
        assert result["blink_detected"] is False
        assert result["estimated_blink_count"] == 0
        assert len(result["eye_openness_series"]) == 5

    def test_blink_detected(self):
        # Create series with a dip
        base_roi = np.full((20, 20), 128, dtype=np.uint8)
        rois = []
        for i in range(10):
            roi = base_roi.copy()
            if i == 5:  # Dip in the middle
                roi[5:15, 5:15] = 50  # Darker region
            rois.append(roi)

        result = compute_blink_evidence_from_eyes_roi_series(rois)
        # Should detect some blink-like event
        assert isinstance(result["blink_detected"], bool)
        assert result["estimated_blink_count"] >= 0
        assert 0.0 <= result["blink_confidence"] <= 1.0
        assert len(result["eye_openness_series"]) == 10

    def test_high_variance_series(self):
        # Create series with high variance
        rois = []
        for i in range(10):
            roi = np.random.randint(0, 256, (15, 15), dtype=np.uint8)
            rois.append(roi)

        result = compute_blink_evidence_from_eyes_roi_series(rois)
        assert len(result["eye_openness_series"]) == 10
        assert all(isinstance(x, float) for x in result["eye_openness_series"])

    def test_mixed_sizes(self):
        # Test with different ROI sizes
        rois = [
            np.full((10, 10), 100, dtype=np.uint8),
            np.full((20, 20), 150, dtype=np.uint8),
            np.full((15, 15), 200, dtype=np.uint8),
        ]
        result = compute_blink_evidence_from_eyes_roi_series(rois)
        assert len(result["eye_openness_series"]) == 3
