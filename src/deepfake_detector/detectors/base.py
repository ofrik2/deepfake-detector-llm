from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Any, Dict

Label = Literal["REAL", "MANIPULATED", "UNCERTAIN"]

@dataclass(frozen=True)
class DetectorResult:
    label: Label
    confidence: Optional[float] = None
    rationale: str = ""
    evidence_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, video_path: str, out_dir: str, config: Optional[Dict[str, Any]] = None) -> DetectorResult:
        """
        Run detection on a video and return a DetectorResult.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the detector.
        """
        pass
