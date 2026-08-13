"""Architecture registry shared by train.py and ds_app.py.

Keeps "which preprocessor/model class goes with which architecture name,
and how to call its forward()" in one place so train.py (which trains and
compares all of them) and ds_app.py (which loads whichever one won) stay
in sync.
"""

from __future__ import annotations

from typing import Any

import torch

from src.model import TabTransformerNet, TitanicNet
from src.preprocessing import TitanicPreprocessor, TitanicTokenizer

ARCH_NAMES = ["mlp", "deep_mlp", "tab_transformer"]


def make_preprocessor(arch: str):
    if arch == "tab_transformer":
        return TitanicTokenizer()
    return TitanicPreprocessor()


def make_model(arch: str, meta: dict[str, Any]) -> torch.nn.Module:
    if arch == "mlp":
        return TitanicNet(n_features=meta["n_features"], hidden_sizes=(32, 16), dropout=0.3)
    if arch == "deep_mlp":
        return TitanicNet(n_features=meta["n_features"], hidden_sizes=(64, 32, 16), dropout=0.3)
    if arch == "tab_transformer":
        return TabTransformerNet(
            n_numeric=meta["n_numeric"],
            cat_cardinalities=meta["cat_cardinalities"],
            d_model=meta.get("d_model", 32),
            n_heads=meta.get("n_heads", 4),
            n_layers=meta.get("n_layers", 2),
            dropout=meta.get("dropout", 0.2),
        )
    raise ValueError(f"Unknown architecture: {arch}")


def forward(model: torch.nn.Module, arch: str, features) -> torch.Tensor:
    """`features` is a single X tensor for mlp/deep_mlp, or an
    (x_num, x_cat) tensor pair for tab_transformer."""
    if arch == "tab_transformer":
        x_num, x_cat = features
        return model(x_num, x_cat)
    return model(features)


def predict_proba(model: torch.nn.Module, arch: str, features, device: torch.device):
    """Run inference and return sigmoid probabilities as a numpy array."""
    model.eval()
    with torch.no_grad():
        if arch == "tab_transformer":
            x_num, x_cat = features
            logits = model(x_num.to(device), x_cat.to(device))
        else:
            logits = model(features.to(device))
    return torch.sigmoid(logits).cpu().numpy()
