from __future__ import annotations
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    """Minimal graph-conv layer: softmax-normalized learnable adjacency over
    a fixed, small node set (EEG channels), then a linear + ReLU projection.
    Avoids a PyTorch Geometric dependency for this small, fixed channel graph.
    """
    def __init__(self, in_dim: int, out_dim: int, num_nodes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.adjacency_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, in_dim)
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)  # (num_nodes, num_nodes)
        aggregated = torch.einsum("ij,bjf->bif", adjacency, x)
        return torch.relu(self.linear(aggregated))
