import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(samples: np.ndarray, sample_rate: int, low: float = 1.0, high: float = 40.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sample_rate
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, samples, axis=0)
