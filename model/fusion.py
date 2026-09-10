from __future__ import annotations
import torch
import torch.nn as nn
from model.encoders import EEGGCNEncoder, BehaviorMLPEncoder

class FusionLSTMClassifier(nn.Module):
    def __init__(
        self, num_channels: int, num_bands: int, num_behavior_feats: int, num_classes: int,
        eeg_embed_dim: int = 16, behavior_embed_dim: int = 16, lstm_hidden_dim: int = 32,
    ):
        super().__init__()
        self.eeg_encoder = EEGGCNEncoder(num_channels, num_bands, out_dim=eeg_embed_dim)
        self.behavior_encoder = BehaviorMLPEncoder(num_behavior_feats, out_dim=behavior_embed_dim)
        self.lstm = nn.LSTM(input_size=eeg_embed_dim + behavior_embed_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, eeg_seq: torch.Tensor, behavior_seq: torch.Tensor) -> torch.Tensor:
        batch, seq_len, num_channels, num_bands = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(batch * seq_len, num_channels, num_bands)
        eeg_embed = self.eeg_encoder(eeg_flat).reshape(batch, seq_len, -1)
        behavior_embed = self.behavior_encoder(behavior_seq)
        fused = torch.cat([eeg_embed, behavior_embed], dim=-1)
        lstm_out, _ = self.lstm(fused)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)
