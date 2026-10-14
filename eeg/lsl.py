"""EEGSource backed by a live Lab Streaming Layer (LSL) stream — the real
headset adapter the design spec reserved the EEGSource seam for.

Any LSL-capable device works (Wearable Sensing DSI via DSI-Streamer, OpenBCI via
its GUI/BrainFlow LSL outlet, Muse via `muselsl`, Emotiv, g.tec, ...): this just
needs a stream of type "EEG". Everything downstream (epoching, features, the
model) is unchanged because this class does the translation, via
preprocessing.realtime:

    native-rate uV  ->  filter  ->  resample to 128 Hz  ->  z-score vs. baseline

Real streams usually carry more than the EEG: DSI-Streamer, for instance, adds a
trigger channel, the common-mode (reference) channel and accelerometer axes next
to the electrodes. Those must never reach the model, so channels are chosen by
LABEL (see `select_channels`): either exactly the ones you name, or — by default
— every channel that isn't a known non-EEG/reference channel.

`pylsl` is imported lazily so the rest of the app (and the test suite) runs
without it installed or without any headset.
"""
from __future__ import annotations
import asyncio
import time
from collections import deque
from typing import AsyncIterator, Optional, Sequence
import numpy as np
from eeg.source import EEGChunk, EEGSource
from common.config import SAMPLE_RATE
from preprocessing.realtime import (
    NOISY_RMS_UV, RunningNormalizer, StreamingConditioner, StreamResampler, channel_quality,
)

# An epoch is flagged as an artifact when any normalised sample exceeds this many
# baseline standard deviations (blinks / jaw clenches / electrode pops).
ARTIFACT_SD = 10.0
QUALITY_WINDOW_SECONDS = 2.0

# Channel labels that are not scalp EEG (case-insensitive): trigger/status lines,
# the DSI common-mode reference, its auxiliary inputs and accelerometer, and the
# ear-reference leads (which read ~0 against themselves).
NON_EEG_LABELS = {
    "TRG", "TRIG", "TRIGGER", "TRIGGERS", "STATUS", "SYSTEM", "COUNTER", "SAMPLE", "SAMPLES", "TIME",
    "DATETIME", "PACKET", "BATTERY", "BAT", "CM", "CMS", "DRL", "GND", "GROUND", "REF", "A1", "A2",
    "X1", "X2", "X3", "AUX", "AUX1", "AUX2", "AUX3", "ACC", "ACCX", "ACCY", "ACCZ", "ACC_X", "ACC_Y", "ACC_Z",
    "GYRO", "GYROX", "GYROY", "GYROZ", "RIGHT AUX", "RIGHTAUX",
}


class LSLStreamNotFound(RuntimeError):
    pass


def select_channels(
    labels: Sequence[str], types: Sequence[str] | None = None, names: Sequence[str] | None = None,
    indices: Sequence[int] | None = None, max_channels: int | None = None,
) -> list[int]:
    """Which stream channel indices carry the EEG the model should see.

    Precedence: explicit `names` (matched case-insensitively against the stream's
    channel labels) > explicit `indices` > automatic (everything not labelled as a
    known non-EEG channel, and not typed as something other than EEG)."""
    total = len(labels)
    if names:
        lookup = {str(l).strip().upper(): i for i, l in enumerate(labels)}
        missing = [n for n in names if n.strip().upper() not in lookup]
        if missing:
            raise LSLStreamNotFound(f"Channel name(s) {missing} not in the stream. Available: {list(labels)}")
        chosen = [lookup[n.strip().upper()] for n in names]
    elif indices is not None:
        bad = [i for i in indices if not 0 <= i < total]
        if bad:
            raise LSLStreamNotFound(f"channel_indices {bad} out of range for a {total}-channel stream.")
        chosen = list(indices)
    else:
        types = types or [""] * total
        chosen = []
        for i, (label, kind) in enumerate(zip(labels, types)):
            if str(label).strip().upper() in NON_EEG_LABELS:
                continue
            if str(kind).strip() and str(kind).strip().upper() not in ("EEG", "EXG"):
                continue
            chosen.append(i)
        if not chosen:   # unlabeled / unexpected stream: fall back to "all of them"
            chosen = list(range(total))
        if max_channels:
            chosen = chosen[:max_channels]
    if not chosen:
        raise LSLStreamNotFound("No EEG channels selected.")
    return chosen


class LSLEEGSource(EEGSource):
    def __init__(
        self, stream_type: str = "EEG", stream_name: Optional[str] = None,
        resolve_timeout: float = 5.0, channel_indices: Optional[list[int]] = None,
        channel_names: Optional[list[str]] = None, max_channels: Optional[int] = None,
        target_rate: int = SAMPLE_RATE, chunk_seconds: float = 0.25,
        notch_hz: Optional[float] = 50.0, target_std: float = 0.5,
        calibration_seconds: float = 30.0, artifact_sd: float = ARTIFACT_SD,
        noisy_rms_uv: float = NOISY_RMS_UV, unit_scale: float = 1.0,
    ):
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.resolve_timeout = resolve_timeout
        self.channel_indices = channel_indices
        self.channel_names_requested = channel_names
        self.max_channels = max_channels
        self.sample_rate = target_rate
        self.chunk_seconds = chunk_seconds
        self.notch_hz = notch_hz
        self.target_std = target_std
        self.calibration_seconds = calibration_seconds
        self.artifact_threshold = artifact_sd * target_std
        self.noisy_rms_uv = noisy_rms_uv
        self.unit_scale = unit_scale          # multiply raw values by this to get microvolts

        self.num_channels = 0
        self.channel_names: list[str] = []    # labels of the channels actually used
        self.all_channel_labels: list[str] = []
        self.native_rate = 0.0
        self.stream_info: dict = {}
        self.unit_warning: str | None = None
        self._inlet = None
        self._conditioner: StreamingConditioner | None = None
        self._resampler = StreamResampler(target_rate)
        self._normalizer: RunningNormalizer | None = None
        self._quality_buffer: deque[np.ndarray] = deque()
        self._quality_samples = 0
        self.latest_quality: list[dict] = []
        self.samples_emitted = 0
        self.last_data_time: float | None = None

    # ------------------------------------------------------------------ connect
    @staticmethod
    def _read_channel_meta(info, total: int) -> tuple[list[str], list[str]]:
        labels, types = [], []
        try:
            node = info.desc().child("channels").child("channel")
            for i in range(total):
                labels.append(node.child_value("label") or f"Ch{i + 1}")
                types.append(node.child_value("type") or "")
                node = node.next_sibling()
        except Exception:  # a stream without channel metadata is still usable
            pass
        if len(labels) != total:
            labels, types = [f"Ch{i + 1}" for i in range(total)], [""] * total
        return labels, types

    def connect(self) -> dict:
        """Resolves the LSL stream and opens an inlet. Blocking (up to
        resolve_timeout seconds) — call it before streaming starts."""
        try:
            import pylsl
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise LSLStreamNotFound("pylsl is not installed (pip install pylsl).") from exc

        if self.stream_name:
            found = pylsl.resolve_byprop("name", self.stream_name, timeout=self.resolve_timeout)
        else:
            found = pylsl.resolve_byprop("type", self.stream_type, timeout=self.resolve_timeout)
        if not found:
            wanted = f"name={self.stream_name!r}" if self.stream_name else f"type={self.stream_type!r}"
            raise LSLStreamNotFound(
                f"No LSL stream with {wanted} found within {self.resolve_timeout:.0f}s. "
                "Is the headset streaming (DSI-Streamer / OpenBCI GUI LSL widget / muselsl)?"
            )

        info = found[0]
        total_channels = info.channel_count()
        native_rate = float(info.nominal_srate())
        if native_rate <= 0:
            raise LSLStreamNotFound("The LSL stream has an irregular sample rate; a regular-rate EEG stream is required.")
        # A resolve result has only the stream header; the per-channel descriptions
        # (labels/types) arrive with the full info of an opened inlet.
        inlet = pylsl.StreamInlet(info, max_buflen=60, processing_flags=pylsl.proc_ALL)
        try:
            labels, types = self._read_channel_meta(inlet.info(timeout=5.0), total_channels)
            self._setup(total_channels, native_rate, labels, types)
        except Exception:
            inlet.close_stream()
            raise
        self._inlet = inlet
        self.stream_info = {
            "name": info.name(), "type": info.type(), "source_id": info.source_id(),
            "total_channels": total_channels, "used_channels": self._indices,
            "channel_names": self.channel_names, "all_channel_labels": labels, "channel_types": types,
            "native_rate": self.native_rate, "model_rate": self.sample_rate,
        }
        return self.stream_info

    def _setup(
        self, total_channels: int, native_rate: float, labels: Sequence[str] | None = None,
        types: Sequence[str] | None = None,
    ) -> None:
        """Channel selection + filter/normaliser construction. Split from
        connect() so the signal path can be exercised without a live stream."""
        self.native_rate = float(native_rate)
        labels = list(labels) if labels else [f"Ch{i + 1}" for i in range(total_channels)]
        self.all_channel_labels = labels
        self._indices = select_channels(
            labels, types, names=self.channel_names_requested, indices=self.channel_indices,
            max_channels=self.max_channels,
        )
        self.channel_names = [labels[i] for i in self._indices]
        self.num_channels = len(self._indices)
        self._conditioner = StreamingConditioner(self.native_rate, notch_hz=self.notch_hz)
        self._resampler = StreamResampler(self.sample_rate)
        self._normalizer = RunningNormalizer(self.num_channels, target_std=self.target_std)

    def close(self) -> None:
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None

    # -------------------------------------------------------------- calibration
    @property
    def calibrating(self) -> bool:
        return self._normalizer is not None and self._normalizer.calibrating

    @property
    def calibration_progress(self) -> float:
        if self._normalizer is None:
            return 0.0
        needed = self.calibration_seconds * self.sample_rate
        return 1.0 if not self._normalizer.calibrating else min(1.0, self._normalizer.count / needed)

    def recalibrate(self, seconds: Optional[float] = None) -> None:
        """Discards the current baseline and learns a new one from the next
        `seconds` of signal — have the participant sit still, eyes open."""
        if self._normalizer is None:
            return
        if seconds is not None:
            self.calibration_seconds = seconds
        self._normalizer.reset()

    def set_baseline(self, baseline: dict) -> None:
        """Re-use a stored per-participant baseline (RunningNormalizer.to_dict)."""
        self._normalizer = RunningNormalizer.from_dict(baseline)

    def baseline(self) -> dict | None:
        return self._normalizer.to_dict() if self._normalizer is not None else None

    # ---------------------------------------------------------------- streaming
    def poll(self) -> Optional[EEGChunk]:
        """Non-blocking: drains whatever the inlet has buffered, conditions it,
        and returns one chunk (or None if nothing new arrived)."""
        if self._inlet is None:
            raise LSLStreamNotFound("poll() before connect().")
        raw, stamps = self._inlet.pull_chunk(timeout=0.0, max_samples=int(self.native_rate * 2))
        if not raw:
            return None
        return self.process_raw(np.asarray(raw, dtype=np.float64), np.asarray(stamps, dtype=np.float64))

    def process_raw(self, raw: np.ndarray, timestamps: np.ndarray) -> Optional[EEGChunk]:
        """The pure signal path (also what tests drive directly): raw native-rate
        samples (n, total_channels) + their timestamps -> one model-rate chunk."""
        assert self._conditioner is not None and self._normalizer is not None
        selected = raw[:, self._indices] * self.unit_scale
        filtered = self._conditioner.process(selected)
        uv = self._resampler.process(timestamps, filtered)
        if uv.shape[0] == 0:
            return None

        self._update_quality(uv)
        if self._normalizer.calibrating:
            self._normalizer.update(uv)
            if self._normalizer.count >= self.calibration_seconds * self.sample_rate:
                self._normalizer.freeze()
        normalised = self._normalizer.transform(uv)

        self.samples_emitted += uv.shape[0]
        self.last_data_time = time.time()
        return EEGChunk(timestamp=time.monotonic(), samples=normalised, uv=uv)

    def _update_quality(self, uv: np.ndarray) -> None:
        self._quality_buffer.append(uv)
        self._quality_samples += uv.shape[0]
        limit = int(QUALITY_WINDOW_SECONDS * self.sample_rate)
        while self._quality_samples - self._quality_buffer[0].shape[0] >= limit:
            self._quality_samples -= self._quality_buffer.popleft().shape[0]
        self.latest_quality = channel_quality(np.vstack(list(self._quality_buffer)), noisy_rms_uv=self.noisy_rms_uv)
        # Every channel "flat" yet clearly varying => almost certainly the stream
        # isn't in microvolts (volts / nanovolts / raw counts), not a dead headset.
        rms = [q["rms"] for q in self.latest_quality]
        if self.samples_emitted > 2 * self.sample_rate and rms and all(q["status"] == "flat" for q in self.latest_quality) and max(rms) > 1e-9:
            self.unit_warning = (
                "Signal is extremely small on every channel — the stream may not be in microvolts "
                "(volts?). Set a unit scale (e.g. 1000000 for volts) when connecting."
            )
        elif all(q["status"] != "flat" for q in self.latest_quality):
            self.unit_warning = None

    async def stream(self) -> AsyncIterator[EEGChunk]:
        while True:
            chunk = self.poll()
            if chunk is not None:
                yield chunk
            await asyncio.sleep(self.chunk_seconds)
