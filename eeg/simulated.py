from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, Optional
import numpy as np
from eeg.source import EEGSource, EEGChunk
from common.config import NUM_CHANNELS, SAMPLE_RATE
from common.states import STATES

STATE_BAND_PARAMS = {
    "Focused":    {"theta": 0.3, "alpha": 0.5},
    "Overloaded": {"theta": 0.9, "alpha": 0.3},
    "Confused":   {"theta": 0.8, "alpha": 0.35},
    "Fatigued":   {"theta": 0.5, "alpha": 0.8},
    "Engaged":    {"theta": 0.4, "alpha": 0.4},
}
assert set(STATE_BAND_PARAMS) == set(STATES)

class SimulatedEEGSource(EEGSource):
    def __init__(
        self,
        num_channels: int = NUM_CHANNELS,
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = 0.25,
        target_state: str = "Focused",
        noise_amplitude: float = 0.2,
        seed: Optional[int] = None,
    ):
        if target_state not in STATE_BAND_PARAMS:
            raise ValueError(f"Unknown state: {target_state}")
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.target_state = target_state
        self.noise_amplitude = noise_amplitude
        self._rng = np.random.default_rng(seed)
        self._t = 0.0

    def set_target_state(self, state: str) -> None:
        if state not in STATE_BAND_PARAMS:
            raise ValueError(f"Unknown state: {state}")
        self.target_state = state

    def generate_chunk(self) -> EEGChunk:
        params = STATE_BAND_PARAMS[self.target_state]
        t = self._t + np.arange(self.chunk_samples) / self.sample_rate
        theta = params["theta"] * np.sin(2 * np.pi * 6.0 * t)
        alpha = params["alpha"] * np.sin(2 * np.pi * 10.0 * t)
        base_signal = theta + alpha
        samples = np.empty((self.chunk_samples, self.num_channels), dtype=np.float64)
        for ch in range(self.num_channels):
            gain = 0.8 + 0.4 * self._rng.random()
            noise = self._rng.normal(0, self.noise_amplitude, size=self.chunk_samples)
            samples[:, ch] = gain * base_signal + noise
        self._t += self.chunk_seconds
        return EEGChunk(timestamp=time.monotonic(), samples=samples)

    async def stream(self) -> AsyncIterator[EEGChunk]:
        while True:
            yield self.generate_chunk()
            await asyncio.sleep(self.chunk_seconds)
