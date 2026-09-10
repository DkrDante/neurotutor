from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator
import numpy as np

@dataclass
class EEGChunk:
    timestamp: float
    samples: np.ndarray  # shape (chunk_samples, num_channels)

class EEGSource(ABC):
    sample_rate: int
    num_channels: int

    @abstractmethod
    async def stream(self) -> AsyncIterator[EEGChunk]:
        """Yield EEGChunk objects indefinitely until the consumer stops iterating."""
        raise NotImplementedError
        yield  # pragma: no cover
