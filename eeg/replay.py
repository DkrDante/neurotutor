"""Play a recorded session back as if it were live.

Two uses: (1) a demo safety net — if the headset, Bluetooth or the room misbehaves
on the day, the live monitor and tutor run unchanged off a real recording;
(2) a credibility check — because the recording knows which condition each
moment belonged to, the /live page can show the model's running estimate right
next to the TRUE recorded condition.

The signal is normalised exactly as at training time (the recording's own
eyes-open baseline), so what is replayed is what the model would have seen live.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import AsyncIterator
import numpy as np
from eeg.recorder import load_recording
from eeg.source import EEGChunk, EEGSource


class ReplayEEGSource(EEGSource):
    def __init__(self, directory: Path, speed: float = 1.0, loop: bool = True, chunk_seconds: float = 0.25):
        from data_gen.from_recordings import fit_baseline, parse_blocks   # lazy: avoids an import cycle
        from preprocessing.realtime import RunningNormalizer

        self.directory = Path(directory)
        rec = load_recording(self.directory)
        if len(rec["eeg"]) == 0:
            raise ValueError(f"Recording {self.directory} contains no EEG samples.")
        meta = rec["meta"]
        self.sample_rate = int(meta["sample_rate"])
        self.num_channels = int(meta["num_channels"])
        self.native_rate = None
        self.speed = max(0.1, float(speed))
        self.loop = loop
        self.chunk_seconds = chunk_seconds
        self.uv = rec["eeg"].astype(np.float64)
        self.times = rec["times"]
        self.blocks = parse_blocks(rec["markers"], float(self.times[-1]))
        baseline, self.baseline_source = fit_baseline(rec, self.blocks)
        self.z = RunningNormalizer.from_dict(baseline).transform(self.uv)
        self.stream_info = {
            "name": f"replay:{self.directory.name}", "type": "EEG (replay)", "participant": meta.get("participant"),
            "baseline": self.baseline_source, "duration_s": float(self.times[-1] - self.times[0]),
        }
        self.cursor = 0
        self.laps = 0
        self.artifact_threshold = 5.0
        self.now_condition: str | None = None

    def _condition_at(self, t: float) -> str | None:
        for b in self.blocks:
            if b.start <= t <= b.end:
                return b.condition
        return None

    @property
    def progress(self) -> float:
        return self.cursor / len(self.z)

    async def stream(self) -> AsyncIterator[EEGChunk]:
        n = max(1, int(self.sample_rate * self.chunk_seconds))
        while True:
            if self.cursor >= len(self.z):
                if not self.loop:
                    return
                self.cursor, self.laps = 0, self.laps + 1
            end = min(len(self.z), self.cursor + n)
            self.now_condition = self._condition_at(float(self.times[end - 1]))
            yield EEGChunk(timestamp=0.0, samples=self.z[self.cursor:end].copy(), uv=self.uv[self.cursor:end].copy())
            self.cursor = end
            await asyncio.sleep(self.chunk_seconds / self.speed)
