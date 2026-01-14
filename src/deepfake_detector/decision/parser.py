from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Label = Literal["REAL", "MANIPULATED", "UNCERTAIN"]


@dataclass(frozen=True)
class Decision:
    label: Label
    reason: str
    raw_text: str


_LABEL_RE = re.compile(
    r"^\s*Label\s*:\s*(REAL|MANIPULATED|UNCERTAIN)\s*$", re.IGNORECASE | re.MULTILINE
)
_REASON_RE = re.compile(r"^\s*Reason\s*:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_llm_output(raw_text: str) -> Decision:
    """
    Parse the strict output format:
      Label: REAL|MANIPULATED|UNCERTAIN
      Reason: <...>

    If parsing fails, returns UNCERTAIN with an explanation.
    """
    label_match = _LABEL_RE.search(raw_text or "")
    reason_match = _REASON_RE.search(raw_text or "")

    if not label_match:
        return Decision(
            label="UNCERTAIN",
            reason="Failed to parse model output label.",
            raw_text=raw_text or "",
        )

    label = label_match.group(1).upper()
    reason = (reason_match.group(1).strip() if reason_match else "No reason provided.").strip()

    # Safety: ensure label is one of allowed
    if label not in {"REAL", "MANIPULATED", "UNCERTAIN"}:
        label = "UNCERTAIN"

    return Decision(label=label, reason=reason, raw_text=raw_text or "")
