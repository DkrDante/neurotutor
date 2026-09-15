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

            # Same submodule-by-submodule approach as the EEG branch above, so the
            # "actual neural network" diagram for the behavior branch (input layer ->
            # hidden layer -> output/embedding layer) can render real per-forward-pass
            # activations at every layer, not just the final embedding's norm.
            behavior_linear1 = self.model.behavior_encoder.net[0]
            behavior_linear2 = self.model.behavior_encoder.net[2]
            behavior_hidden = torch.relu(behavior_linear1(behavior_tensor))
            behavior_embed = behavior_linear2(behavior_hidden)
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
        # Both GCN layers have their own learned adjacency (see model/graph.py) — the
        # live diagram used to only ever surface gcn1's, silently dropping gcn2's own
        # graph structure (and the fact that it's degenerate: gcn1's near-uniform
        # output collapses every channel to nearly identical values, zeroing gcn2's
        # gradient and leaving its adjacency at its zero-initialization — see
        # model/evidence.py / model/MODEL_EVIDENCE.md for the full writeup).
        gcn1_adjacency = torch.softmax(self.model.eeg_encoder.gcn1.adjacency_logits, dim=-1)
        gcn2_adjacency = torch.softmax(self.model.eeg_encoder.gcn2.adjacency_logits, dim=-1)
        classifier = self.model.classifier

        return {
            "state": STATES[best_idx],
            "confidence": float(probs[best_idx]),
            "probs": prob_dict,
            "gcn1_node_activity": gcn1_out[last_step].norm(dim=-1).tolist(),
            "gcn2_node_activity": gcn2_out[last_step].norm(dim=-1).tolist(),
            "gcn1_adjacency": gcn1_adjacency.tolist(),
            "gcn2_adjacency": gcn2_adjacency.tolist(),
            "eeg_embedding_norm": float(eeg_embed[0, -1].norm().item()),
            "behavior_embedding_norm": float(behavior_embed[0, -1].norm().item()),
            "lstm_hidden_norm": float(last_hidden[0].norm().item()),
            # Full per-unit LSTM hidden state, plus the real classifier weight matrix
            # (5 states x 32 hidden units) — lets the live diagram draw the LSTM ->
            # output hop as real weighted edges (same "fixed weight x current
            # activation" treatment as the behavior branch's edges below) instead of
            # a single opaque "LSTM" blob feeding generic, weightless flow lines.
            "lstm_hidden_activity": last_hidden[0].tolist(),
            "classifier_weight": classifier.weight.detach().tolist(),  # (num_states, lstm_hidden_dim)
            "behavior_activity": [float(v) for v in behavior_seq[-1]],
            # Real per-forward-pass activations for the behavior MLP's hidden and
            # output layers, plus its two fixed, learned weight matrices — enough to
            # render the behavior branch as an actual layered neural-network diagram
            # (input -> hidden -> output) rather than a single opaque "embedding" node.
            "behavior_hidden_activity": behavior_hidden[0, -1].tolist(),
            "behavior_output_activity": behavior_embed[0, -1].tolist(),
            "behavior_w1": behavior_linear1.weight.detach().tolist(),  # (hidden_dim, in_dim)
            "behavior_w2": behavior_linear2.weight.detach().tolist(),  # (out_dim, hidden_dim)
        }
