"""Streaming conditioning for a real headset (preprocessing/realtime.py)."""
from __future__ import annotations
import numpy as np
from preprocessing.realtime import (
    RunningNormalizer, StreamResampler, StreamingConditioner, channel_quality,
)


def _power_at(x: np.ndarray, rate: float, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    return float(spectrum[np.argmin(np.abs(freqs - freq))])


def test_conditioner_removes_dc_and_mains_but_keeps_alpha():
    rate = 250.0
    t = np.arange(int(rate * 12)) / rate
    raw = 4000 + 20 * np.sin(2 * np.pi * 10 * t) + 40 * np.sin(2 * np.pi * 50 * t)
    out = StreamingConditioner(rate, notch_hz=50.0).process(raw[:, None])[:, 0]
    settled = out[int(rate * 4):]  # skip the filter's start-up
    assert abs(settled.mean()) < 1.0                                   # DC gone
    assert _power_at(settled, rate, 50) < 0.05 * _power_at(raw[int(rate * 4):], rate, 50)
    assert 0.8 < _power_at(settled, rate, 10) / _power_at(raw[int(rate * 4):], rate, 10) < 1.2


def test_conditioner_is_chunk_invariant():
    rate = 250.0
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 3)) * 10 + 500
    whole = StreamingConditioner(rate).process(x)
    parts = StreamingConditioner(rate)
    chunked = np.vstack([parts.process(x[i:i + 37]) for i in range(0, 1000, 37)])
    assert np.allclose(whole, chunked, atol=1e-9)


def test_resampler_hits_target_rate_and_preserves_a_sine_across_chunks():
    native, target = 250.0, 128.0
    t = np.arange(int(native * 10)) / native
    x = np.sin(2 * np.pi * 8 * t)[:, None]
    resampler = StreamResampler(target)
    pieces = [resampler.process(t[i:i + 25], x[i:i + 25]) for i in range(0, len(t), 25)]
    out = np.vstack(pieces)
    assert abs(len(out) - 10 * target) <= 2
    grid_t = t[0] + np.arange(len(out)) / target
    assert np.max(np.abs(out[:, 0] - np.sin(2 * np.pi * 8 * grid_t))) < 0.02


def test_resampler_resumes_after_a_dropout_instead_of_interpolating_across_it():
    resampler = StreamResampler(128.0)
    t1 = np.arange(0, 1, 1 / 250.0)
    resampler.process(t1, np.zeros((len(t1), 1)))
    t2 = 30.0 + np.arange(0, 1, 1 / 250.0)  # a 29 s hole
    out = resampler.process(t2, np.zeros((len(t2), 1)))
    assert len(out) < 128 * 2  # not 29 s of fabricated samples


def test_normalizer_matches_numpy_stats_and_freezes():
    rng = np.random.default_rng(1)
    data = rng.normal(5, 3, size=(2000, 4))
    norm = RunningNormalizer(4, target_std=0.5)
    for i in range(0, 2000, 128):
        norm.update(data[i:i + 128])
    assert np.allclose(norm.mean, data.mean(axis=0))
    assert np.allclose(norm.std, data.std(axis=0, ddof=1))
    norm.freeze()
    before = norm.mean.copy()
    norm.update(rng.normal(100, 1, size=(500, 4)))  # ignored once frozen
    assert np.array_equal(norm.mean, before)
    out = norm.transform(data)
    assert np.allclose(out.std(axis=0), 0.5, atol=0.01) and np.allclose(out.mean(axis=0), 0, atol=0.01)


def test_normalizer_round_trips_through_dict():
    norm = RunningNormalizer(2)
    norm.update(np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]]))
    norm.freeze()
    again = RunningNormalizer.from_dict(norm.to_dict())
    x = np.array([[2.0, 3.0]])
    assert np.allclose(norm.transform(x), again.transform(x)) and not again.calibrating


def test_channel_quality_flags_flat_noisy_and_ok():
    rng = np.random.default_rng(2)
    window = np.column_stack([
        rng.normal(0, 20, 256),    # healthy
        np.full(256, 3.0),         # lifted electrode
        rng.normal(0, 400, 256),   # motion / muscle
    ])
    assert [q["status"] for q in channel_quality(window)] == ["ok", "flat", "noisy"]
