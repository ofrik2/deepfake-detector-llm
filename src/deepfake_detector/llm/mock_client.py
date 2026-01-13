from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .client_base import LLMClient, LLMResponse


def _extract_float(prompt: str, key: str) -> Optional[float]:
    """
    Extract a float from lines like:
      - Mouth-region motion mean abs diff: 14.12
    """
    pattern = re.compile(rf"{re.escape(key)}\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    m = pattern.search(prompt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


@dataclass
class MockLLMClient(LLMClient):
    model_name: str = "mock-llm"

    def generate(self, *, prompt: str, image_paths: Optional[List[str]] = None) -> LLMResponse:
        # Pull evidence values if present
        mouth = _extract_float(prompt, "Mouth-region motion mean abs diff")
        eyes = _extract_float(prompt, "Eye-region motion mean abs diff")

        label = "UNCERTAIN"
        reason = "Insufficient evidence in prompt to make a strong determination."

        if mouth is not None and eyes is not None:
            # Simple interpretable heuristic:
            # - If mouth >> eyes: suggests speaking with unusually static eyes (suspicious)
            # - If mouth ~ eyes: could be normal blinking/head motion
            if eyes < 2.0 and mouth > 6.0:
                label = "MANIPULATED"
                reason = "Mouth motion is present while eye-region motion is extremely low, which may indicate unnatural static eyes."
            elif mouth / max(eyes, 1e-6) >= 2.5:
                label = "MANIPULATED"
                reason = "Mouth motion is much higher than eye-region motion, which can indicate unnatural eye behavior."
            elif 0.7 <= (mouth / max(eyes, 1e-6)) <= 1.5:
                label = "REAL"
                reason = "Mouth and eye-region motion are of similar magnitude, consistent with natural speaking and blinking/head motion."
            else:
                label = "UNCERTAIN"
                reason = "Motion distribution is not clearly indicative of real or manipulated content."

        text = f"Label: {label}\nReason: {reason}"
        return LLMResponse(raw_text=text, model_name=self.model_name, usage=None)
