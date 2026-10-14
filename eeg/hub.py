"""EEGHub: one process-wide owner of the (single) live EEG source.

An LSL inlet can only be drained by one reader, but several things want the
same live signal at once: the tutor's WebSocket session (to feed its epoch
buffer), the /live monitoring page, and the session recorder. The hub owns the
source, pumps it in one background task, and fans each chunk out to any number
of subscribers — plus a short rolling window for live analysis and an optional
SessionRecorder.

The source is whatever was connected: an LSLEEGSource (real headset / the fake
LSL stream) or a SimulatedEEGSource (demo / no hardware).
"""
from __future__ import annotations
import asyncio
import time
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Optional
import numpy as np
from eeg.recorder import DEFAULT_ROOT, SessionRecorder
from eeg.source import EEGChunk, EEGSource

RING_SECONDS = 24.0  # enough for 5 x 4 s sequence steps (+ headroom)
SUBSCRIBER_QUEUE_SIZE = 256


class EEGHub:
    def __init__(self, recordings_root: Path = DEFAULT_ROOT):
        self.recordings_root = Path(recordings_root)
        self.source: Optional[EEGSource] = None
        self.kind: Optional[str] = None  # "lsl" | "sim"
        self.connected_at: Optional[float] = None
        self.chunks_received = 0
        self.last_chunk_wall: Optional[float] = None
        self.error: Optional[str] = None      # why the pump stopped, if it crashed
        self.recorder: Optional[SessionRecorder] = None
        self.current_block: Optional[dict] = None
        self._task: Optional[asyncio.Task] = None
        self._subscribers: set[asyncio.Queue] = set()
        self._event_subscribers: set[asyncio.Queue] = set()
        self._ring: deque[np.ndarray] = deque()
        self._ring_samples = 0

    # ----------------------------------------------------------------- lifecycle
    @property
    def connected(self) -> bool:
        return self.source is not None and self._task is not None and not self._task.done()

    async def connect(self, source: EEGSource, kind: str) -> None:
        await self.disconnect()
        self.source, self.kind = source, kind
        self.connected_at = time.time()
        self.chunks_received = 0
        self.last_chunk_wall = None
        self.error = None
        self._ring.clear()
        self._ring_samples = 0
        self._task = asyncio.create_task(self._pump())

    async def disconnect(self) -> None:
        if self.recorder is not None and self.recorder.active:
            self.recorder.stop()
        self.recorder = None
        self.current_block = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        closer = getattr(self.source, "close", None)
        if callable(closer):
            closer()
        self.source, self.kind, self.connected_at = None, None, None

    async def _pump(self) -> None:
        assert self.source is not None
        try:
            await self._pump_chunks()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a crashed source must be visible, not a silent "disconnected"
            self.error = f"{type(exc).__name__}: {exc}"
            if self.recorder is not None and self.recorder.active:
                self.recorder.add_marker("source_error", error=self.error)

    async def _pump_chunks(self) -> None:
        assert self.source is not None
        async for chunk in self.source.stream():
            wall = time.time()
            self.chunks_received += 1
            self.last_chunk_wall = wall
            self._push_ring(chunk.samples)
            if self.recorder is not None and self.recorder.active:
                self.recorder.write_chunk(wall, chunk)
            for queue in list(self._subscribers):
                if queue.full():
                    try:
                        queue.get_nowait()  # drop the oldest rather than stall the pump
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(chunk)

    def _push_ring(self, samples: np.ndarray) -> None:
        self._ring.append(samples)
        self._ring_samples += samples.shape[0]
        limit = int(RING_SECONDS * (self.source.sample_rate if self.source else 128))
        while self._ring and self._ring_samples - self._ring[0].shape[0] >= limit:
            self._ring_samples -= self._ring.popleft().shape[0]

    # --------------------------------------------------------------- subscribers
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def subscribe_events(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._event_subscribers.add(queue)
        return queue

    def unsubscribe_events(self, queue: asyncio.Queue) -> None:
        self._event_subscribers.discard(queue)

    def publish_event(self, event: dict) -> None:
        """Tutor sessions report their updates here so /live can show the
        adaptation (difficulty, action, correctness) next to the signal."""
        for queue in list(self._event_subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    # -------------------------------------------------------------------- window
    def window(self, seconds: float) -> np.ndarray:
        """The most recent `seconds` of (normalised) signal, shape (n, channels);
        fewer rows if less has arrived."""
        if not self._ring or self.source is None:
            return np.zeros((0, getattr(self.source, "num_channels", 0)))
        data = np.vstack(list(self._ring))
        count = int(seconds * self.source.sample_rate)
        return data[-count:]

    def quality(self) -> list[dict]:
        latest = getattr(self.source, "latest_quality", None)
        if latest:
            return latest
        channels = getattr(self.source, "num_channels", 0)
        if self.kind in ("sim", "replay"):
            return [{"rms": 0.0, "status": "simulated" if self.kind == "sim" else "replay"} for _ in range(channels)]
        return [{"rms": 0.0, "status": "no_data"} for _ in range(channels)]

    # ----------------------------------------------------------------- recording
    def start_recording(self, participant: str, label: str = "") -> SessionRecorder:
        if self.source is None:
            raise RuntimeError("No EEG source connected.")
        if self.recorder is not None and self.recorder.active:
            raise RuntimeError("Already recording.")
        baseline_fn = getattr(self.source, "baseline", None)
        self.recorder = SessionRecorder(
            participant=participant, sample_rate=self.source.sample_rate,
            num_channels=self.source.num_channels, root=self.recordings_root, label=label,
            source_info={"kind": self.kind, **getattr(self.source, "stream_info", {})},
            baseline=baseline_fn() if callable(baseline_fn) else None,
        )
        self.recorder.add_marker("recording_start", participant=participant, label=label)
        return self.recorder

    def stop_recording(self, attempts: list[dict] | None = None) -> Optional[Path]:
        if self.recorder is None or not self.recorder.active:
            return None
        self.recorder.add_marker("recording_stop")
        directory = self.recorder.stop(attempts=attempts)
        self.current_block = None
        return directory

    def add_marker(self, kind: str, **data) -> dict:
        marker = {"t": time.time(), "kind": kind, **data}
        if kind == "block_start":
            self.current_block = {k: v for k, v in marker.items() if k != "kind"}
        elif kind == "block_end":
            self.current_block = None
        if self.recorder is not None and self.recorder.active:
            marker = self.recorder.add_marker(kind, **data)
        return marker

    # -------------------------------------------------------------------- status
    def status(self) -> dict:
        source = self.source
        now = time.time()
        recorder = self.recorder if self.recorder is not None and self.recorder.active else None
        return {
            "connected": self.connected,
            "error": self.error,
            "kind": self.kind,
            "num_channels": getattr(source, "num_channels", 0),
            "channel_names": list(getattr(source, "channel_names", []) or []),
            "unit_warning": getattr(source, "unit_warning", None),
            "sample_rate": getattr(source, "sample_rate", 0),
            "native_rate": getattr(source, "native_rate", None),
            "stream_info": getattr(source, "stream_info", {}),
            "calibrating": bool(getattr(source, "calibrating", False)),
            "calibration_progress": getattr(source, "calibration_progress", 1.0),
            "seconds_since_data": (now - self.last_chunk_wall) if self.last_chunk_wall else None,
            "chunks_received": self.chunks_received,
            "uptime": (now - self.connected_at) if self.connected_at else 0.0,
            "recording": None if recorder is None else {
                "participant": recorder.participant, "directory": str(recorder.directory),
                "seconds": now - recorder.started_at, "samples": recorder.samples_written,
            },
            "current_block": self.current_block,
            "replay": None if not hasattr(source, "now_condition") else {
                "condition": source.now_condition, "progress": source.progress, "laps": source.laps,
                "speed": source.speed, "participant": source.stream_info.get("participant"),
            },
        }


class HubSubscriberSource(EEGSource):
    """Adapts the hub to the EEGSource interface, so a TutorSession can consume
    the shared live signal exactly as it would a private simulator."""

    def __init__(self, hub: EEGHub):
        if hub.source is None:
            raise RuntimeError("No EEG source is connected to the hub.")
        self._hub = hub
        self.sample_rate = hub.source.sample_rate
        self.num_channels = hub.source.num_channels
        threshold = getattr(hub.source, "artifact_threshold", None)
        if threshold is not None:
            self.artifact_threshold = threshold

    async def stream(self) -> AsyncIterator[EEGChunk]:
        queue = self._hub.subscribe()
        try:
            while True:
                yield await queue.get()
        finally:
            self._hub.unsubscribe(queue)


def collect_attempts(store, start: float, end: float) -> list[dict]:
    """All chess attempts logged between two wall-clock times, across sessions —
    copied next to the recording so it is self-contained."""
    rows = []
    for session in store.list_sessions():
        for attempt in store.get_session_attempts(session["session_id"]):
            if start <= attempt["created_at"] <= end:
                rows.append({"session_id": session["session_id"], **attempt})
    return sorted(rows, key=lambda r: r["created_at"])
