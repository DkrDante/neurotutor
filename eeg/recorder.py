"""Session recorder: writes a headset recording to disk in a form the dataset
builder (data_gen/from_recordings.py) can turn into labelled training examples
with no manual annotation.

Layout (one directory per recording):

    recordings/<participant>/<YYYYmmdd-HHMMSS>[-<label>]/
        meta.json        participant, sample rate, channels, source info, start/end
        eeg_uv.f32       float32 (n_samples, n_channels), row-major — conditioned
                         microvolts at the model rate (BEFORE per-person
                         normalisation, so the baseline can be applied offline)
        eeg_time.f64     float64 (n_samples,) wall-clock seconds per sample
        markers.jsonl    one JSON object per line: {"t": wall_time, "kind": ..., ...}
        attempts.csv     the chess attempts logged during the recording window
                         (copied from the SQLite store at stop time)

Everything is aligned on wall-clock time (`time.time()`), so EEG samples, block
markers and chess attempts (storage.db stamps `created_at` with the same clock)
can be joined with a simple interval lookup.
"""
from __future__ import annotations
import csv
import json
import re
import time
from pathlib import Path
import numpy as np
from eeg.source import EEGChunk

DEFAULT_ROOT = Path("recordings")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()) or "unnamed"


class SessionRecorder:
    def __init__(
        self, participant: str, sample_rate: int, num_channels: int,
        root: Path = DEFAULT_ROOT, label: str = "", source_info: dict | None = None,
        baseline: dict | None = None,
    ):
        self.participant = participant
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        started = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(started))
        name = f"{stamp}-{_slug(label)}" if label else stamp
        self.directory = Path(root) / _slug(participant) / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started_at = started
        self.ended_at: float | None = None
        self.samples_written = 0
        self.chunks_without_uv = 0
        self._meta_extra = {"source": source_info or {}, "baseline": baseline}
        self._eeg = open(self.directory / "eeg_uv.f32", "ab")
        self._time = open(self.directory / "eeg_time.f64", "ab")
        self._markers = open(self.directory / "markers.jsonl", "a")
        self._last_time = 0.0
        self._write_meta()

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def write_chunk(self, wall_time: float, chunk: EEGChunk) -> None:
        # Prefer the unnormalised microvolt signal; the simulator has none, so
        # its (already unit-scale) samples are stored instead and flagged.
        data = chunk.uv if chunk.uv is not None else chunk.samples
        if chunk.uv is None:
            self.chunks_without_uv += 1
        n = data.shape[0]
        # `wall_time` is when the chunk arrived, i.e. ~ the time of its LAST sample.
        times = wall_time - (n - 1 - np.arange(n)) / self.sample_rate
        times = np.maximum(times, self._last_time + 1e-9) if self.samples_written else times
        self._last_time = float(times[-1])
        self._eeg.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())
        self._time.write(np.ascontiguousarray(times, dtype=np.float64).tobytes())
        self.samples_written += n

    def add_marker(self, kind: str, _t: float | None = None, **data) -> dict:
        """_t overrides the wall-clock stamp (used when synthesising a recording)."""
        marker = {"t": time.time() if _t is None else _t, "kind": kind, **data}
        self._markers.write(json.dumps(marker) + "\n")
        self._markers.flush()
        return marker

    def _write_meta(self) -> None:
        meta = {
            "participant": self.participant, "sample_rate": self.sample_rate,
            "num_channels": self.num_channels, "started_at": self.started_at,
            "ended_at": self.ended_at, "samples": self.samples_written,
            "eeg_dtype": "float32", "eeg_units": "microvolts (conditioned, pre-normalisation)",
            "uv_available": self.chunks_without_uv == 0,
            **self._meta_extra,
        }
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2))

    def stop(self, attempts: list[dict] | None = None) -> Path:
        if not self.active:
            return self.directory
        self.ended_at = time.time()
        for handle in (self._eeg, self._time, self._markers):
            handle.flush()
            handle.close()
        if attempts is not None:
            self._write_attempts(attempts)
        self._write_meta()
        return self.directory

    def _write_attempts(self, attempts: list[dict]) -> None:
        fields = [
            "session_id", "mode", "puzzle_id", "correct", "time_to_move", "eval_loss", "puzzle_rating",
            "predicted_state", "confidence", "difficulty_delta", "show_hint", "pacing_delay",
            "sense_to_adapt_latency", "created_at",
        ]
        with open(self.directory / "attempts.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for attempt in attempts:
                writer.writerow(attempt)


def load_recording(directory: Path) -> dict:
    """Reads a recording back: {"meta", "eeg" (n, ch) float32, "times" (n,), "markers", "attempts"}."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    channels = meta["num_channels"]
    eeg = np.fromfile(directory / "eeg_uv.f32", dtype=np.float32).reshape(-1, channels)
    times = np.fromfile(directory / "eeg_time.f64", dtype=np.float64)
    n = min(len(eeg), len(times))  # a crash mid-write can leave the two a chunk apart
    markers = []
    marker_file = directory / "markers.jsonl"
    if marker_file.exists():
        markers = [json.loads(line) for line in marker_file.read_text().splitlines() if line.strip()]
    attempts = []
    attempts_file = directory / "attempts.csv"
    if attempts_file.exists():
        with open(attempts_file, newline="") as f:
            for row in csv.DictReader(f):
                attempts.append({
                    **row,
                    "correct": row["correct"] in ("True", "1", "true"),
                    "show_hint": row["show_hint"] in ("True", "1", "true"),
                    "time_to_move": float(row["time_to_move"]), "eval_loss": float(row["eval_loss"]),
                    "puzzle_rating": int(float(row["puzzle_rating"])), "created_at": float(row["created_at"]),
                    "confidence": float(row["confidence"]),
                })
    return {"meta": meta, "eeg": eeg[:n], "times": times[:n], "markers": markers, "attempts": attempts}
