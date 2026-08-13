"""PyTorch model definitions for Titanic survival classification.

Three architectures are trained and compared by train.py:
  - `mlp`           TitanicNet(32, 16)   — small baseline MLP
  - `deep_mlp`       TitanicNet(64, 32, 16) — more capacity
  - `tab_transformer` TabTransformerNet   — per-column feature tokenization
                       + a small Transformer encoder (FT-Transformer style)

All output a single logit (use BCEWithLogitsLoss / sigmoid at inference).
"""

from __future__ import annotations

import torch
from torch import nn


class TitanicNet(nn.Module):
    """MLP binary classifier for a flat, one-hot-encoded feature vector.

    Tabular, low-dimensional input (~20-25 scaled/one-hot features) with
    only ~700 training rows, so a compact MLP with dropout is enough
    capacity without overfitting badly.
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


class TabTransformerNet(nn.Module):
    """Small tabular Transformer (FT-Transformer / TabTransformer style).

    Each numeric column is linearly projected to a `d_model`-dim token
    (its own nn.Linear(1, d_model)); each categorical column gets its own
    nn.Embedding. All per-column tokens plus a learnable [CLS] token are
    concatenated into a sequence and passed through a standard
    TransformerEncoder; the [CLS] output is the pooled representation used
    for classification.

    Note: with ~700 training rows this has noticeably more parameters than
    the MLP for the same information, so it's included as a genuine
    architecture comparison rather than an expected winner — attention over
    ~20 tokens on a dataset this small tends to be capacity the data can't
    support, which is itself a useful, realistic result to report.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_numeric = n_numeric
        self.numeric_tokenizers = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_numeric)]
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.shape[0]
        num_tokens = [
            tok(x_num[:, i : i + 1]) for i, tok in enumerate(self.numeric_tokenizers)
        ]
        cat_tokens = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)
        ]
        tokens = torch.stack(num_tokens + cat_tokens, dim=1)  # (batch, n_tokens, d_model)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder(tokens)
        cls_out = self.dropout(self.norm(encoded[:, 0]))
        return self.head(cls_out).squeeze(-1)  # logits, shape (batch,)
