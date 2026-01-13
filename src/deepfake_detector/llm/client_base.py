from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class LLMResponse:
    """
    Standard response object returned by any LLM client implementation.
    """
    raw_text: str
    model_name: str
    usage: Optional[dict] = None  # keep flexible (tokens, cost, etc.)


class LLMClient(Protocol):
    """
    Protocol / interface for LLM clients.

    We keep this minimal: a client takes prompt text and optional image paths,
    and returns a text response.

    Different model providers can implement this interface.
    """

    def generate(
        self,
        *,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> LLMResponse:
        ...
