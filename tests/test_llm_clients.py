import pytest

from src.deepfake_detector.llm.client_base import LLMResponse
from src.deepfake_detector.llm.mock_client import MockLLMClient, _extract_float


class TestExtractFloat:
    def test_extract_existing_value(self):
        prompt = "Mouth-region motion mean abs diff: 14.12\nEye-region motion mean abs diff: 2.5"
        assert _extract_float(prompt, "Mouth-region motion mean abs diff") == 14.12
        assert _extract_float(prompt, "Eye-region motion mean abs diff") == 2.5

    def test_extract_nonexistent_key(self):
        prompt = "Some other text: 1.0"
        assert _extract_float(prompt, "Missing key") is None

    def test_extract_invalid_number(self):
        prompt = "Key: not_a_number"
        assert _extract_float(prompt, "Key") is None

    def test_extract_with_extra_spaces(self):
        prompt = "Key   :   123.45   "
        assert _extract_float(prompt, "Key") == 123.45

    def test_extract_integer(self):
        prompt = "Key: 42"
        assert _extract_float(prompt, "Key") == 42.0


class TestMockLLMClient:
    def test_init_default_model_name(self):
        client = MockLLMClient()
        assert client.model_name == "mock-llm"

    def test_init_custom_model_name(self):
        client = MockLLMClient(model_name="custom-mock")
        assert client.model_name == "custom-mock"

    def test_generate_no_motion_data(self):
        client = MockLLMClient()
        prompt = "Some prompt without motion data"
        response = client.generate(prompt=prompt)

        assert isinstance(response, LLMResponse)
        assert response.model_name == "mock-llm"
        assert response.usage is None
        assert "UNCERTAIN" in response.raw_text
        assert "Insufficient evidence" in response.raw_text

    def test_generate_manipulated_high_mouth_low_eyes(self):
        client = MockLLMClient()
        prompt = "Mouth-region motion mean abs diff: 10.0\nEye-region motion mean abs diff: 1.0"
        response = client.generate(prompt=prompt)

        assert "MANIPULATED" in response.raw_text
        assert "Mouth motion is present while eye-region motion is extremely low" in response.raw_text

    def test_generate_manipulated_ratio_based(self):
        client = MockLLMClient()
        prompt = "Mouth-region motion mean abs diff: 5.0\nEye-region motion mean abs diff: 1.0"
        response = client.generate(prompt=prompt)

        assert "MANIPULATED" in response.raw_text
        assert "Mouth motion is much higher than eye-region motion" in response.raw_text

    def test_generate_real_similar_motion(self):
        client = MockLLMClient()
        prompt = "Mouth-region motion mean abs diff: 3.0\nEye-region motion mean abs diff: 2.5"
        response = client.generate(prompt=prompt)

        assert "REAL" in response.raw_text
        assert "Mouth and eye-region motion are of similar magnitude" in response.raw_text

    def test_generate_uncertain_motion_ratio(self):
        client = MockLLMClient()
        prompt = "Mouth-region motion mean abs diff: 2.0\nEye-region motion mean abs diff: 1.0"
        response = client.generate(prompt=prompt)

        assert "UNCERTAIN" in response.raw_text
        assert "Motion distribution is not clearly indicative" in response.raw_text

    def test_generate_with_image_paths(self):
        client = MockLLMClient()
        prompt = "Test prompt"
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        response = client.generate(prompt=prompt, image_paths=image_paths)

        # Mock client doesn't use image_paths, but should not crash
        assert isinstance(response, LLMResponse)

    def test_generate_empty_image_paths(self):
        client = MockLLMClient()
        prompt = "Test prompt"
        response = client.generate(prompt=prompt, image_paths=[])

        assert isinstance(response, LLMResponse)

    def test_generate_none_image_paths(self):
        client = MockLLMClient()
        prompt = "Test prompt"
        response = client.generate(prompt=prompt, image_paths=None)

        assert isinstance(response, LLMResponse)

    def test_response_format(self):
        client = MockLLMClient()
        prompt = "Mouth-region motion mean abs diff: 1.0\nEye-region motion mean abs diff: 1.0"
        response = client.generate(prompt=prompt)

        # Should follow the expected format
        lines = response.raw_text.strip().split('\n')
        assert len(lines) == 2
        assert lines[0].startswith("Label: ")
        assert lines[1].startswith("Reason: ")

    def test_edge_case_zero_eyes_motion(self):
        client = MockLLMClient()
        prompt = "Mouth-region motion mean abs diff: 1.0\nEye-region motion mean abs diff: 0.0"
        response = client.generate(prompt=prompt)

        # Should not crash due to division by zero
        assert isinstance(response, LLMResponse)
        assert "MANIPULATED" in response.raw_text