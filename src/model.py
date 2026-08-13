"""PyTorch model definition for Titanic survival classification."""

from __future__ import annotations

import torch
from torch import nn


class TitanicNet(nn.Module):
    """Small MLP binary classifier.

    Tabular, low-dimensional input (~15 one-hot/scaled features) with only
    ~600 training rows, so a compact 2-hidden-layer MLP with dropout is
    enough capacity without overfitting badly. Outputs a single logit
    (use BCEWithLogitsLoss / sigmoid at inference).
    """

    def __init__(self, n_features: int, hidden_sizes=(32, 16), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits, shape (batch,)
