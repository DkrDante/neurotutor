from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class NpzSequenceDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.X_eeg = torch.tensor(data["X_eeg"], dtype=torch.float32)
        self.X_behavior = torch.tensor(data["X_behavior"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_eeg[idx], self.X_behavior[idx], self.y[idx]
