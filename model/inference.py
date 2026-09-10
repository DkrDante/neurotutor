from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES

class StatePredictor:
    def __init__(self, checkpoint_path: Path):
        self.model = FusionLSTMClassifier(
            num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
            num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, eeg_seq: np.ndarray, behavior_seq: np.ndarray):
        eeg_tensor = torch.tensor(eeg_seq, dtype=torch.float32).unsqueeze(0)
        behavior_tensor = torch.tensor(behavior_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(eeg_tensor, behavior_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        best_idx = int(torch.argmax(probs).item())
        prob_dict = {state: float(probs[i]) for i, state in enumerate(STATES)}
        return STATES[best_idx], float(probs[best_idx]), prob_dict
