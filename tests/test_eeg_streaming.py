"""Regression tests for the live EEG path.

The original bug: web.server recorded exactly ONE 0.25s chunk per puzzle, which
EpochBuffer then zero-padded to the full 4.0s epoch. At serving time the GCN
branch therefore saw ~99.97% zeros, while training used full 4.0s chunks. These
tests pin down the fix (continuous streaming into the buffer) by comparing
serving-time band-power magnitudes against the training-time magnitudes.
"""
from __future__ import annotations
import asyncio
import numpy as np
from common.config import NUM_CHANNELS, SAMPLE_RATE, EPOCH_SAMPLES, EPOCH_SECONDS
from eeg.simulated import SimulatedEEGSource
from preprocessing.epoching import EpochBuffer
from preprocessing.filters import bandpass_filter
from preprocessing.features import extract_node_features


def _training_time_features(state: str = "Focused", seed: int = 0) -> np.ndarray:
    """Exactly what data_gen.generate_dataset.generate_examples produces per step."""
    src = SimulatedEEGSource(
        num_channels=NUM_CHANNELS, sample_rate=SAMPLE_RATE,
        chunk_seconds=EPOCH_SAMPLES / SAMPLE_RATE, target_state=state, seed=seed,
    )
    chunk = src.generate_chunk()
    return extract_node_features(bandpass_filter(chunk.samples, SAMPLE_RATE), SAMPLE_RATE)


def _serving_time_features(num_chunks: int, state: str = "Focused", seed: int = 0) -> np.ndarray:
    """What TutorSession sees after `num_chunks` streamed chunks land in the buffer."""
    src = SimulatedEEGSource(target_state=state, seed=seed)  # default 0.25s chunks
    buf = EpochBuffer(num_channels=src.num_channels, sample_rate=src.sample_rate)
    for _ in range(num_chunks):
        buf.add_chunk(src.generate_chunk())
    epoch = buf.extract_epoch()
    return extract_node_features(bandpass_filter(epoch.samples, src.sample_rate), src.sample_rate)


def test_single_chunk_epoch_is_near_zero_compared_to_training():
    """The OLD behavior — this documents the bug the fix removes."""
    training = _training_time_features()
    one_chunk = _serving_time_features(num_chunks=1)
    # A single 0.25s chunk zero-padded to 4.0s loses ~99.9% of the band power.
    assert one_chunk.sum() < 0.01 * training.sum()


def test_streamed_epoch_band_power_matches_training_magnitudes():
    """The FIXED behavior: ~4s of continuously streamed signal, no meaningful padding."""
    chunks_for_full_epoch = int(EPOCH_SECONDS / SimulatedEEGSource().chunk_seconds)  # 16
    training = _training_time_features()
    streamed = _serving_time_features(num_chunks=chunks_for_full_epoch)

    # Not near-zero in absolute terms.
    assert streamed.sum() > 0.1
    assert np.all(streamed >= 0.0)

    # And in the same ballpark as the training-time features, per band.
    training_bands = training.sum(axis=0)
    streamed_bands = streamed.sum(axis=0)
    for band_idx in range(training_bands.shape[0]):
        ratio = streamed_bands[band_idx] / training_bands[band_idx]
        assert 0.4 < ratio < 2.5, (
            f"band {band_idx}: streamed/training power ratio {ratio:.3f} out of range"
        )


def test_stream_interface_fills_buffer_over_time(monkeypatch):
    """Drive EEGSource.stream() the way web.server's background task does."""
    async def _instant(_delay):  # skip stream()'s real-time pacing
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)

    src = SimulatedEEGSource(target_state="Focused", seed=3)
    buf = EpochBuffer(num_channels=src.num_channels, sample_rate=src.sample_rate)

    async def pump(n: int) -> None:
        count = 0
        async for chunk in src.stream():
            buf.add_chunk(chunk)
            count += 1
            if count >= n:
                return

    asyncio.run(pump(int(EPOCH_SECONDS / src.chunk_seconds)))
    epoch = buf.extract_epoch()
    assert epoch.samples.shape == (EPOCH_SAMPLES, NUM_CHANNELS)
    # Every row carries real signal — nothing was zero-padded.
    assert np.count_nonzero(np.abs(epoch.samples).sum(axis=1)) == EPOCH_SAMPLES

    features = extract_node_features(bandpass_filter(epoch.samples, src.sample_rate), src.sample_rate)
    assert features.sum() > 0.1
