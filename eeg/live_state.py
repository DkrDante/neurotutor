"""Continuous, EEG-only state estimates for the /live monitoring page.

The tutor itself only predicts when a move is submitted (it needs that attempt's
behaviour features). A monitoring page wants a running readout, so this runs an
EEG-only model (behaviour input zeroed — the same trick as the ablation's
eeg_only variant) over the hub's rolling window, once per refresh.

Which checkpoint is used is a deliberate, visible choice: a model trained on
REAL recordings (`live_eeg.pt`, produced by the real-data trainer) wins if it
exists; otherwise the synthetic-data eeg_only ablation model is used and every
estimate is flagged `preliminary` so the UI says so rather than implying it was
validated on a headset.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from common.config import BANDS, BAND_NAMES, EPOCH_SECONDS, NUM_BEHAVIOR_FEATS, SEQ_LEN
from model.inference import StatePredictor
from preprocessing.features import extract_node_features
from preprocessing.filters import bandpass_filter

REAL_CHECKPOINT = Path("model/checkpoints/live_eeg.pt")
SYNTHETIC_CHECKPOINT = Path("model/checkpoints/ablation-eeg_only.pt")


def band_summary(window: np.ndarray, sample_rate: int) -> dict | None:
    """Per-channel and average band power for the latest epoch-length window,
    absolute and relative (share of theta+alpha+beta)."""
    epoch_samples = int(EPOCH_SECONDS * sample_rate)
    if window.shape[0] < epoch_samples:
        return None
    features = extract_node_features(bandpass_filter(window[-epoch_samples:], sample_rate), sample_rate)
    totals = np.maximum(features.sum(axis=1, keepdims=True), 1e-12)
    relative = features / totals
    return {
        "bands": BAND_NAMES,
        "ranges_hz": {name: list(rng) for name, rng in BANDS.items()},
        "absolute": features.tolist(),
        "relative": relative.tolist(),
        "mean_absolute": features.mean(axis=0).tolist(),
        "mean_relative": relative.mean(axis=0).tolist(),
    }


class LiveStateEstimator:
    def __init__(self, real_checkpoint: Path = REAL_CHECKPOINT, synthetic_checkpoint: Path = SYNTHETIC_CHECKPOINT):
        self.predictor: StatePredictor | None = None
        self.label = "no model available"
        self.preliminary = True
        for path, label, preliminary in (
            (real_checkpoint, "EEG-only model trained on real recordings", False),
            (synthetic_checkpoint, "synthetic-data EEG-only model — preliminary", True),
        ):
            if Path(path).exists():
                self.predictor = StatePredictor(Path(path))
                self.label, self.preliminary = label, preliminary
                break

    @property
    def available(self) -> bool:
        return self.predictor is not None

    @property
    def num_channels(self) -> int | None:
        return self.predictor.num_channels if self.predictor is not None else None

    def estimate(self, window: np.ndarray, sample_rate: int) -> dict | None:
        """window: the hub's most recent signal (n, channels). Returns None until
        at least one full epoch has arrived; earlier sequence steps with no data
        yet are zero-padded, exactly as TutorSession pads a young session."""
        if self.predictor is None or window.shape[1] != self.predictor.num_channels:
            return None   # no model for this headset's channel count yet — the page says so
        epoch_samples = int(EPOCH_SECONDS * sample_rate)
        if window.shape[0] < epoch_samples:
            return None
        steps = []
        for k in range(SEQ_LEN - 1, -1, -1):  # oldest -> newest
            end = window.shape[0] - k * epoch_samples
            start = end - epoch_samples
            if start < 0:
                steps.append(np.zeros((window.shape[1], len(BAND_NAMES)), dtype=np.float32))
            else:
                steps.append(extract_node_features(bandpass_filter(window[start:end], sample_rate), sample_rate).astype(np.float32))
        eeg_seq = np.stack(steps)
        behavior_seq = np.zeros((SEQ_LEN, NUM_BEHAVIOR_FEATS), dtype=np.float32)
        state, confidence, probs = self.predictor.predict(eeg_seq, behavior_seq)
        return {
            "state": state, "confidence": confidence, "probs": probs,
            "model": self.label, "preliminary": self.preliminary,
            "history_seconds": min(window.shape[0], epoch_samples * SEQ_LEN) / sample_rate,
        }
