"""Streaming signal conditioning for a real EEG headset.

The simulator emits clean, unit-scale, already-128Hz samples. A real headset
(streamed over LSL) does not: values are in microvolts with a large DC offset
and slow drift, mains hum (50/60 Hz) rides on top, the native rate is usually
250/256 Hz rather than the model's 128 Hz, and each person's amplitude scale
differs. This module turns that into the same kind of signal the model was
built around, causally and chunk-by-chunk (no look-ahead, so it works live):

    raw uV @ native rate
      -> StreamingConditioner   (highpass 0.5 Hz, mains notch, lowpass 45 Hz)
      -> StreamResampler        (native rate -> model rate, by timestamp)
      -> [uV @ model rate]      <- also what the recorder stores
      -> RunningNormalizer      (per-channel z-score vs. a resting baseline)
      -> unit-scale samples the existing pipeline consumes

Nothing here imports pylsl, so all of it is unit-testable without a headset.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

# Quality thresholds on the conditioned (0.5-45 Hz) signal, in microvolts RMS.
# Real scalp EEG is typically ~5-50 uV RMS; a lifted electrode reads ~0, and a
# poor contact / motion / EMG burst reads far above physiological levels.
FLAT_RMS_UV = 0.5
NOISY_RMS_UV = 150.0


class StreamingConditioner:
    """Causal per-channel filter chain with persistent state across chunks."""

    def __init__(
        self, sample_rate: float, notch_hz: float | None = 50.0,
        highpass_hz: float = 0.5, lowpass_hz: float = 45.0,
    ):
        nyquist = sample_rate / 2.0
        sections = [butter(2, highpass_hz / nyquist, btype="highpass", output="sos")]
        if notch_hz and notch_hz < nyquist * 0.98:
            b, a = iirnotch(notch_hz, Q=30.0, fs=sample_rate)
            sections.append(tf2sos(b, a))
        # Never ask for a lowpass at/above Nyquist (e.g. a 64 Hz headset).
        lowpass = min(lowpass_hz, nyquist * 0.9)
        sections.append(butter(4, lowpass / nyquist, btype="lowpass", output="sos"))
        self.sos = np.vstack(sections)
        self._zi: np.ndarray | None = None

    def process(self, samples: np.ndarray) -> np.ndarray:
        """samples: (n, channels) -> filtered (n, channels), same shape."""
        if samples.shape[0] == 0:
            return samples
        if self._zi is None:
            # Start from the steady state of the first sample so the highpass
            # doesn't ring on the (large) DC offset a real headset sits at.
            zi = sosfilt_zi(self.sos)  # (sections, 2)
            self._zi = zi[:, :, None] * samples[0][None, None, :]
        filtered, self._zi = sosfilt(self.sos, samples, axis=0, zi=self._zi)
        return filtered


class StreamResampler:
    """Linear-interpolation resampler onto a uniform grid, driven by the
    samples' own timestamps (so LSL timestamp jitter and a native rate that
    isn't an integer multiple of the target rate both just work). Callers must
    low-pass first (StreamingConditioner does) so the decimation can't alias."""

    MAX_GAP_SECONDS = 2.0

    def __init__(self, target_rate: float):
        self.target_rate = float(target_rate)
        self._next_t: float | None = None
        self._last_t: float | None = None
        self._last_x: np.ndarray | None = None

    def process(self, timestamps: np.ndarray, samples: np.ndarray) -> np.ndarray:
        num_channels = samples.shape[1] if samples.ndim == 2 else 0
        if len(timestamps) == 0:
            return np.zeros((0, num_channels))

        ts, xs = np.asarray(timestamps, dtype=np.float64), samples
        if self._last_t is not None:
            ts = np.concatenate([[self._last_t], ts])
            xs = np.vstack([self._last_x, xs])
        # Samples must be time-ordered for np.interp.
        order = np.argsort(ts, kind="stable")
        ts, xs = ts[order], xs[order]

        step = 1.0 / self.target_rate
        if self._next_t is None:
            self._next_t = float(ts[0])
        elif ts[-1] - self._next_t > self.MAX_GAP_SECONDS:
            # Dropout / stalled stream: resume at the present rather than
            # fabricating seconds of interpolated signal across the gap.
            self._next_t = float(ts[-1]) - 0.25

        count = int(np.floor((ts[-1] - self._next_t) / step + 1e-9)) + 1
        self._last_t, self._last_x = float(ts[-1]), xs[-1:].copy()
        if count <= 0:
            return np.zeros((0, num_channels))

        grid = self._next_t + np.arange(count) * step
        self._next_t = float(grid[-1] + step)
        out = np.empty((count, num_channels), dtype=np.float64)
        for ch in range(num_channels):
            out[:, ch] = np.interp(grid, ts, xs[:, ch])
        return out


class RunningNormalizer:
    """Per-channel z-score against a baseline, learned online (Welford/Chan
    parallel update) while `calibrating`, then frozen. Output is rescaled to
    `target_std` so band powers land in the same magnitude range the model was
    trained on (the simulator's per-channel std is roughly 0.4-0.6)."""

    def __init__(self, num_channels: int, target_std: float = 0.5):
        self.num_channels = num_channels
        self.target_std = target_std
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.num_channels)
        self._m2 = np.zeros(self.num_channels)
        self.calibrating = True
        self._frozen_std: np.ndarray | None = None

    def update(self, samples: np.ndarray) -> None:
        if not self.calibrating or samples.shape[0] == 0:
            return
        n_b = samples.shape[0]
        mean_b = samples.mean(axis=0)
        m2_b = ((samples - mean_b) ** 2).sum(axis=0)
        n_a = self.count
        total = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * n_b / total
        self._m2 = self._m2 + m2_b + delta ** 2 * n_a * n_b / total
        self.count = total

    def freeze(self) -> None:
        self._frozen_std = self.std
        self.calibrating = False

    @property
    def std(self) -> np.ndarray:
        if self._frozen_std is not None:
            return self._frozen_std
        if self.count < 2:
            return np.ones(self.num_channels)
        return np.sqrt(self._m2 / (self.count - 1))

    def transform(self, samples: np.ndarray) -> np.ndarray:
        std = np.maximum(self.std, 1e-6)
        return (samples - self.mean) / std * self.target_std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "target_std": self.target_std, "count": int(self.count),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunningNormalizer":
        norm = cls(len(data["mean"]), target_std=data.get("target_std", 0.5))
        norm.mean = np.asarray(data["mean"], dtype=np.float64)
        norm.count = int(data.get("count", 0))
        norm._frozen_std = np.asarray(data["std"], dtype=np.float64)
        norm.calibrating = False
        return norm


def channel_quality(window_uv: np.ndarray, noisy_rms_uv: float = NOISY_RMS_UV, flat_rms_uv: float = FLAT_RMS_UV) -> list[dict]:
    """Per-channel signal-quality verdict from a conditioned uV window
    (n, channels): 'flat' (electrode lifted / no signal), 'noisy' (motion,
    muscle, bad contact) or 'ok'. RMS is reported so the UI can show it."""
    if window_uv.shape[0] < 2:
        return [{"rms": 0.0, "status": "no_data"} for _ in range(window_uv.shape[1])]
    rms = np.sqrt(np.mean((window_uv - window_uv.mean(axis=0)) ** 2, axis=0))
    out = []
    for value in rms:
        status = "flat" if value < flat_rms_uv else "noisy" if value > noisy_rms_uv else "ok"
        out.append({"rms": float(value), "status": status})
    return out
