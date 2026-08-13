"""Permutation feature importance for the trained Titanic classifier.

Model-agnostic: shuffles one engineered feature at a time in the held-out
validation set and measures the resulting drop in accuracy, averaged over
several repeats. Operates on the fitted preprocessor's `engineer()` /
`transform_engineered()` split, so it works identically regardless of which
architecture won (mlp / deep_mlp / tab_transformer) — no gradients or
model-specific internals required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.architectures import predict_proba
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _predict(model, arch, preprocessor, engineered_df: pd.DataFrame, device):
    if arch == "tab_transformer":
        num, cat = preprocessor.transform_engineered(engineered_df)
        features = (torch.from_numpy(num), torch.from_numpy(cat))
    else:
        X = preprocessor.transform_engineered(engineered_df)
        features = torch.from_numpy(X)
    return predict_proba(model, arch, features, device)


def permutation_importance(
    model, arch: str, preprocessor, val_df: pd.DataFrame, device, n_repeats: int = 15, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    engineered = preprocessor.engineer(val_df)
    y_true = engineered[TARGET].to_numpy()

    baseline_probs = _predict(model, arch, preprocessor, engineered, device)
    baseline_acc = float(((baseline_probs > 0.5).astype(int) == y_true).mean())

    importances: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        drops = []
        for _ in range(n_repeats):
            shuffled = engineered.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            probs = _predict(model, arch, preprocessor, shuffled, device)
            acc = ((probs > 0.5).astype(int) == y_true).mean()
            drops.append(baseline_acc - acc)
        importances[col] = float(np.mean(drops))

    ranked = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    return {"baseline_accuracy": baseline_acc, "importances": ranked}
