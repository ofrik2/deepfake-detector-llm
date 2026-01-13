from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .client_base import LLMClient, LLMResponse


@dataclass
class GeminiLLMClient(LLMClient):
    """
    Gemini client placeholder.

    We keep this file in place so the project structure is complete.
    The real implementation depends on:
      - which Gemini SDK/version you use
      - how your assignment expects secrets to be provided (.env vs env vars)
      - whether you send images as bytes or paths

    For now: raise a clear error so you can keep developing with MockLLMClient.
    """
    model_name: str = "gemini"

    def generate(self, *, prompt: str, image_paths: Optional[List[str]] = None) -> LLMResponse:
        raise RuntimeError(
            "GeminiLLMClient is not configured yet. "
            "Use MockLLMClient for now, or implement Gemini calls when ready."
        )
