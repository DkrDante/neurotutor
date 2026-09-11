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
        result = self.predict_with_internals(eeg_seq, behavior_seq)
        return result["state"], result["confidence"], result["probs"]

    def predict_with_internals(self, eeg_seq: np.ndarray, behavior_seq: np.ndarray) -> dict:
        """Same prediction as predict(), plus real internal activations/weights for the
        live network-activity visualization: per-channel GCN node magnitudes, the
        model's learned (softmax) channel-adjacency weights, and embedding/hidden-state
        norms. Nothing here is synthetic — it's the actual forward pass, computed one
        submodule at a time instead of via model.forward() so these values are visible.
        """
        eeg_tensor = torch.tensor(eeg_seq, dtype=torch.float32).unsqueeze(0)
        behavior_tensor = torch.tensor(behavior_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            batch, seq_len, num_channels, num_bands = eeg_tensor.shape
            eeg_flat = eeg_tensor.reshape(batch * seq_len, num_channels, num_bands)

            gcn1_out = self.model.eeg_encoder.gcn1(eeg_flat)
            gcn2_out = self.model.eeg_encoder.gcn2(gcn1_out)
            eeg_embed = gcn2_out.mean(dim=1).reshape(batch, seq_len, -1)

            behavior_embed = self.model.behavior_encoder(behavior_tensor)
            fused = torch.cat([eeg_embed, behavior_embed], dim=-1)
            lstm_out, _ = self.model.lstm(fused)
            last_hidden = lstm_out[:, -1, :]
            logits = self.model.classifier(last_hidden)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        best_idx = int(torch.argmax(probs).item())
        prob_dict = {state: float(probs[i]) for i, state in enumerate(STATES)}

        # gcn1_out/gcn2_out have shape (batch*seq_len, num_channels, hidden_dim); with
        # batch=1 the last row is the most recent timestep's per-channel node vectors.
        last_step = seq_len - 1
        adjacency = torch.softmax(self.model.eeg_encoder.gcn1.adjacency_logits, dim=-1)

        return {
            "state": STATES[best_idx],
            "confidence": float(probs[best_idx]),
            "probs": prob_dict,
            "gcn1_node_activity": gcn1_out[last_step].norm(dim=-1).tolist(),
            "gcn2_node_activity": gcn2_out[last_step].norm(dim=-1).tolist(),
            "gcn_adjacency": adjacency.tolist(),
            "eeg_embedding_norm": float(eeg_embed[0, -1].norm().item()),
            "behavior_embedding_norm": float(behavior_embed[0, -1].norm().item()),
            "lstm_hidden_norm": float(last_hidden[0].norm().item()),
            "behavior_activity": [float(v) for v in behavior_seq[-1]],
        }
