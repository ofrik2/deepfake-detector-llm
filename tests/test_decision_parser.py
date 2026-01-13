import pytest

from src.deepfake_detector.decision.parser import Decision, parse_llm_output


class TestParseLlmOutput:
    def test_parse_valid_real(self):
        raw_text = "Label: REAL\nReason: The video appears authentic."
        decision = parse_llm_output(raw_text)

        assert decision.label == "REAL"
        assert decision.reason == "The video appears authentic."
        assert decision.raw_text == raw_text

    def test_parse_valid_manipulated(self):
        raw_text = "Label: MANIPULATED\nReason: Evidence suggests manipulation."
        decision = parse_llm_output(raw_text)

        assert decision.label == "MANIPULATED"
        assert decision.reason == "Evidence suggests manipulation."
        assert decision.raw_text == raw_text

    def test_parse_valid_uncertain(self):
        raw_text = "Label: UNCERTAIN\nReason: Insufficient evidence."
        decision = parse_llm_output(raw_text)

        assert decision.label == "UNCERTAIN"
        assert decision.reason == "Insufficient evidence."
        assert decision.raw_text == raw_text

    def test_parse_case_insensitive_label(self):
        raw_text = "label: real\nreason: test"
        decision = parse_llm_output(raw_text)

        assert decision.label == "REAL"

    def test_parse_with_extra_whitespace(self):
        raw_text = "  Label:   REAL   \n  Reason:   Test reason   "
        decision = parse_llm_output(raw_text)

        assert decision.label == "REAL"
        assert decision.reason == "Test reason"

    def test_parse_multiline_reason(self):
        raw_text = "Label: MANIPULATED\nReason: This is a\nmultiline reason\nwith more text."
        decision = parse_llm_output(raw_text)

        assert decision.label == "MANIPULATED"
        assert decision.reason == "This is a\nmultiline reason\nwith more text."

    def test_parse_missing_label(self):
        raw_text = "Reason: Some reason"
        decision = parse_llm_output(raw_text)

        assert decision.label == "UNCERTAIN"
        assert "Failed to parse model output label" in decision.reason
        assert decision.raw_text == raw_text

    def test_parse_missing_reason(self):
        raw_text = "Label: REAL"
        decision = parse_llm_output(raw_text)

        assert decision.label == "REAL"
        assert decision.reason == "No reason provided."
        assert decision.raw_text == raw_text

    def test_parse_invalid_label(self):
        raw_text = "Label: INVALID\nReason: Test"
        decision = parse_llm_output(raw_text)

        assert decision.label == "UNCERTAIN"
        assert decision.reason == "Failed to parse model output label."
        assert decision.raw_text == raw_text

    def test_parse_empty_string(self):
        raw_text = ""
        decision = parse_llm_output(raw_text)

        assert decision.label == "UNCERTAIN"
        assert "Failed to parse model output label" in decision.reason
        assert decision.raw_text == ""

    def test_parse_none_input(self):
        decision = parse_llm_output(None)

        assert decision.label == "UNCERTAIN"
        assert "Failed to parse model output label" in decision.reason
        assert decision.raw_text == ""

    def test_parse_label_with_extra_text(self):
        raw_text = "Some text\nLabel: REAL\nMore text\nReason: Test"
        decision = parse_llm_output(raw_text)

        assert decision.label == "REAL"
        assert decision.reason == "Test"

    def test_parse_reason_with_colon(self):
        raw_text = "Label: MANIPULATED\nReason: This: is a reason with: colons"
        decision = parse_llm_output(raw_text)

        assert decision.label == "MANIPULATED"
        assert decision.reason == "This: is a reason with: colons"

    def test_parse_multiple_labels(self):
        raw_text = "Label: REAL\nLabel: MANIPULATED\nReason: Test"
        decision = parse_llm_output(raw_text)

        # Should take the first match
        assert decision.label == "REAL"
        assert decision.reason == "Test"

    def test_parse_label_only_uppercase(self):
        raw_text = "LABEL: REAL\nREASON: Test"
        decision = parse_llm_output(raw_text)

        assert decision.label == "REAL"
        assert decision.reason == "Test"

    def test_parse_with_preceding_text(self):
        raw_text = "Some analysis here\nLabel: UNCERTAIN\nReason: Not enough info"
        decision = parse_llm_output(raw_text)

        assert decision.label == "UNCERTAIN"
        assert decision.reason == "Not enough info"