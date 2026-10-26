"""In-memory audio captured by an input adapter."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordedAudio:
    """One-dimensional mono float32 samples and their actual sample rate."""

    samples: Any
    sample_rate: int
