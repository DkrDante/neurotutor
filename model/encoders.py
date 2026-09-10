from __future__ import annotations
import torch
import torch.nn as nn
from model.graph import GCNLayer

class EEGGCNEncoder(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, num_nodes)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, in_dim) -> (batch, out_dim)
        h = self.gcn1(x)
        h = self.gcn2(h)
        return h.mean(dim=1)

class BehaviorMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
