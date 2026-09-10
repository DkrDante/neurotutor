from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from eeg.source import EEGChunk
from common.config import EPOCH_SECONDS, ARTIFACT_THRESHOLD

@dataclass
class Epoch:
    samples: np.ndarray  # shape (num_samples, num_channels)
    is_artifact: bool

class EpochBuffer:
    def __init__(self, num_channels: int, sample_rate: int, epoch_seconds: float = EPOCH_SECONDS, artifact_threshold: float = ARTIFACT_THRESHOLD):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.epoch_samples = int(sample_rate * epoch_seconds)
        self.artifact_threshold = artifact_threshold
        self._buffer: list[np.ndarray] = []

    def add_chunk(self, chunk: EEGChunk) -> None:
        self._buffer.append(chunk.samples)

    def extract_epoch(self) -> Epoch:
        if self._buffer:
            all_samples = np.concatenate(self._buffer, axis=0)
        else:
            all_samples = np.zeros((0, self.num_channels))
        self._buffer = []

        if len(all_samples) >= self.epoch_samples:
            windowed = all_samples[-self.epoch_samples:]
        elif len(all_samples) > 0:
            pad = np.zeros((self.epoch_samples - len(all_samples), self.num_channels))
            windowed = np.concatenate([pad, all_samples], axis=0)
        else:
            windowed = np.zeros((self.epoch_samples, self.num_channels))

        is_artifact = bool(np.max(np.abs(windowed)) > self.artifact_threshold)
        return Epoch(samples=windowed, is_artifact=is_artifact)
