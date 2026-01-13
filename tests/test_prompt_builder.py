import pytest

from src.deepfake_detector.prompts.prompt_builder import build_prompt_text


class TestBuildPromptText:
    def test_basic_prompt_structure(self):
        llm_input = {
            "sampling": {
                "sampled_frame_count": 12,
                "sampling_mode": "uniform",
                "fps": 30.0,
                "duration_seconds": 10.0
            },
            "evidence": {
                "roi_method": "haar_face_per_frame",
                "global_motion_mean": 1.5,
                "mouth_motion_mean": 2.0,
                "eyes_motion_mean": 0.5,
                "face_bbox_none_ratio": 0.1,
                "face_bbox_center_jitter_mean": 0.05,
                "face_bbox_size_jitter_mean": 0.02,
                "boundary_face_ratio_mean": 1.2,
                "boundary_bg_ratio_mean": 0.8,
                "boundary_face_over_bg_mean": 1.5,
                "boundary_face_minus_bg_mean": 0.4,
                "eyes_motion_p95": 1.0,
                "eyes_motion_max_over_mean": 2.5,
                "mouth_motion_p95": 3.0,
                "mouth_motion_max_over_mean": 1.8,
                "eye_openness_range": 0.8,
                "blink_dip_fraction": 0.1,
                "estimated_blink_count": 2,
                "blink_confidence": 0.7,
                "blink_like_events": 1,
                "notes": ["Test note 1", "Test note 2"]
            },
            "keyframes": [
                {"timestamp_seconds": 0.0},
                {"timestamp_seconds": 5.0},
                {"timestamp_seconds": 10.0}
            ]
        }

        prompt = build_prompt_text(llm_input)

        # Check basic structure
        assert "You are analyzing a short video" in prompt
        assert "Context:" in prompt
        assert "Extracted evidence" in prompt
        assert "Task:" in prompt
        assert "Output format" in prompt

        # Check specific values are included
        assert "Sampled frames: 12" in prompt
        assert "mode=uniform" in prompt
        assert "FPS: 30.0" in prompt
        assert "duration_seconds: 10.0" in prompt
        assert "Keyframe timestamps (seconds): [0.0, 5.0, 10.0]" in prompt
        assert "ROI method: haar_face_per_frame" in prompt
        assert "Global motion mean abs diff: 1.5" in prompt
        assert "Mouth-region motion mean abs diff: 2.0" in prompt
        assert "Eye-region motion mean abs diff: 0.5" in prompt
        assert "Face detection missing ratio: 0.1" in prompt
        assert "Face-boundary edge ratio mean (ring/inner): 1.2" in prompt
        assert "Blink dip fraction: 0.1" in prompt
        assert "Test note 1" in prompt
        assert "Test note 2" in prompt

    def test_minimal_input(self):
        llm_input = {
            "sampling": {},
            "evidence": {},
            "keyframes": []
        }

        prompt = build_prompt_text(llm_input)

        assert "Sampled frames: None" in prompt
        assert "Keyframes: none provided" in prompt
        assert "ROI method: unknown" in prompt

    def test_no_keyframes(self):
        llm_input = {
            "sampling": {"sampled_frame_count": 5},
            "evidence": {},
            "keyframes": []
        }

        prompt = build_prompt_text(llm_input)

        assert "Keyframes: none provided" in prompt

    def test_missing_evidence_fields(self):
        llm_input = {
            "sampling": {},
            "evidence": {
                "roi_method": "test",
                "notes": ["Note"]
            },
            "keyframes": []
        }

        prompt = build_prompt_text(llm_input)

        # Should not crash, just skip missing fields
        assert "ROI method: test" in prompt
        assert "Note" in prompt

    def test_boundary_evidence_old_format(self):
        llm_input = {
            "sampling": {},
            "evidence": {
                "boundary_edge_ratio_mean": 1.0  # Old field name
            },
            "keyframes": []
        }

        prompt = build_prompt_text(llm_input)

        assert "Face-boundary edge ratio mean (ring/inner): 1.0" in prompt

    def test_boundary_evidence_new_format(self):
        llm_input = {
            "sampling": {},
            "evidence": {
                "boundary_face_ratio_mean": 1.2,
                "boundary_bg_ratio_mean": 0.8,
                "boundary_face_over_bg_mean": 1.5,
                "boundary_face_minus_bg_mean": 0.4
            },
            "keyframes": []
        }

        prompt = build_prompt_text(llm_input)

        assert "Face-boundary edge ratio mean (ring/inner): 1.2" in prompt
        assert "Background-boundary edge ratio mean: 0.8" in prompt
        assert "Boundary face_over_bg mean: 1.5" in prompt
        assert "Boundary face_minus_bg mean: 0.4" in prompt

    def test_prompt_contains_instructions(self):
        llm_input = {"sampling": {}, "evidence": {}, "keyframes": []}
        prompt = build_prompt_text(llm_input)

        assert "Decide whether the video is more likely REAL or MANIPULATED" in prompt
        assert "Label: REAL|MANIPULATED|UNCERTAIN" in prompt
        assert "Reason: <1-5 sentences>" in prompt

    def test_prompt_formatting(self):
        llm_input = {"sampling": {}, "evidence": {}, "keyframes": []}
        prompt = build_prompt_text(llm_input)

        # Should be properly formatted with newlines
        lines = prompt.split('\n')
        assert len(lines) > 10  # Should have multiple lines
        assert any(line.strip() == "" for line in lines)  # Should have empty lines for formatting
