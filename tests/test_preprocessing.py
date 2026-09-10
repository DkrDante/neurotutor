import numpy as np
from preprocessing.filters import bandpass_filter
from preprocessing.features import band_power, extract_node_features
from preprocessing.epoching import EpochBuffer
from eeg.source import EEGChunk
from common.config import BAND_NAMES

def test_bandpass_filter_attenuates_out_of_band():
    sample_rate = 128
    t = np.arange(sample_rate * 2) / sample_rate
    low_freq = np.sin(2 * np.pi * 0.2 * t)
    in_band = np.sin(2 * np.pi * 10 * t)
    signal = (low_freq + in_band).reshape(-1, 1)
    filtered = bandpass_filter(signal, sample_rate)
    assert np.std(filtered) < np.std(signal)

def test_band_power_higher_in_target_band():
    sample_rate = 128
    t = np.arange(sample_rate * 4) / sample_rate
    theta_signal = np.sin(2 * np.pi * 6 * t)
    gamma_signal = np.sin(2 * np.pi * 45 * t)
    theta_power = band_power(theta_signal, sample_rate, (4, 8))
    gamma_power_in_theta_band = band_power(gamma_signal, sample_rate, (4, 8))
    assert theta_power > gamma_power_in_theta_band

def test_extract_node_features_shape():
    sample_rate = 128
    epoch = np.random.default_rng(0).normal(size=(512, 8))
    features = extract_node_features(epoch, sample_rate)
    assert features.shape == (8, len(BAND_NAMES))

def test_epoch_buffer_pads_short_input():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=4.0)
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=np.ones((32, 8))))
    epoch = buf.extract_epoch()
    assert epoch.samples.shape == (512, 8)
    assert np.all(epoch.samples[:480] == 0)
    assert np.all(epoch.samples[480:] == 1)

def test_epoch_buffer_truncates_long_input():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=1.0)
    long_input = np.arange(256 * 8).reshape(256, 8).astype(float)
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=long_input))
    epoch = buf.extract_epoch()
    assert epoch.samples.shape == (128, 8)
    assert np.array_equal(epoch.samples, long_input[-128:])

def test_epoch_buffer_flags_artifact():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=1.0, artifact_threshold=5.0)
    samples = np.zeros((128, 8))
    samples[0, 0] = 100.0
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=samples))
    epoch = buf.extract_epoch()
    assert epoch.is_artifact is True

def test_epoch_buffer_clears_after_extract():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=1.0)
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=np.ones((128, 8))))
    buf.extract_epoch()
    second = buf.extract_epoch()
    assert np.all(second.samples == 0)
