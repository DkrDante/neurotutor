import numpy as np
import pytest
from scipy.signal import welch
from eeg.simulated import SimulatedEEGSource

def _band_power(samples_1d, sample_rate, band):
    freqs, psd = welch(samples_1d, fs=sample_rate, nperseg=min(256, len(samples_1d)))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[mask], freqs[mask]))

def test_generate_chunk_shape():
    src = SimulatedEEGSource(num_channels=8, sample_rate=128, chunk_seconds=0.25, seed=1)
    chunk = src.generate_chunk()
    assert chunk.samples.shape == (32, 8)

def test_overloaded_has_more_theta_power_than_focused():
    focused = SimulatedEEGSource(target_state="Focused", chunk_seconds=4.0, seed=1)
    overloaded = SimulatedEEGSource(target_state="Overloaded", chunk_seconds=4.0, seed=1)
    f_chunk = focused.generate_chunk()
    o_chunk = overloaded.generate_chunk()
    f_power = np.mean([_band_power(f_chunk.samples[:, ch], 128, (4, 8)) for ch in range(8)])
    o_power = np.mean([_band_power(o_chunk.samples[:, ch], 128, (4, 8)) for ch in range(8)])
    assert o_power > f_power

def test_set_target_state_changes_state():
    src = SimulatedEEGSource(target_state="Focused", seed=1)
    src.set_target_state("Fatigued")
    assert src.target_state == "Fatigued"

def test_invalid_state_raises():
    with pytest.raises(ValueError):
        SimulatedEEGSource(target_state="NotAState")
