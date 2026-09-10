import numpy as np
from scipy.signal import welch
from common.config import BANDS, BAND_NAMES

def band_power(channel_samples: np.ndarray, sample_rate: int, band: tuple[float, float]) -> float:
    nperseg = min(256, len(channel_samples))
    freqs, psd = welch(channel_samples, fs=sample_rate, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[mask], freqs[mask]))

def extract_node_features(epoch: np.ndarray, sample_rate: int) -> np.ndarray:
    """epoch: shape (num_samples, num_channels). Returns (num_channels, len(BAND_NAMES))."""
    num_channels = epoch.shape[1]
    features = np.zeros((num_channels, len(BAND_NAMES)), dtype=np.float64)
    for ch in range(num_channels):
        for b_idx, band_name in enumerate(BAND_NAMES):
            features[ch, b_idx] = band_power(epoch[:, ch], sample_rate, BANDS[band_name])
    return features
