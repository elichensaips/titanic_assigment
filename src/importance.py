"""Permutation feature importance for the trained Titanic classifiers.

Model-agnostic: shuffles one engineered feature at a time in the held-out
validation set and measures the resulting drop in accuracy, averaged over
several repeats. Operates on the fitted preprocessor's `engineer()` /
`transform_engineered()` split, so the core `permutation_importance()`
function only needs a `predict_fn(engineered_df) -> probabilities` callable
— it works identically for the PyTorch architectures (mlp / deep_mlp /
tab_transformer, via `torch_predict_fn`) and for the plain scikit-learn-API
baseline models trained by train_baselines.py (logistic regression, KNN,
SVM, random forest, gradient boosting, ...), with no gradients or
model-specific internals required either way.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.architectures import predict_proba
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def torch_predict_fn(model, arch: str, preprocessor, device) -> PredictFn:
    """predict_fn for a PyTorch model: engineered_df -> P(survived)."""

    def predict_fn(engineered_df: pd.DataFrame) -> np.ndarray:
        if arch == "tab_transformer":
            num, cat = preprocessor.transform_engineered(engineered_df)
            features = (torch.from_numpy(num), torch.from_numpy(cat))
        else:
            X = preprocessor.transform_engineered(engineered_df)
            features = torch.from_numpy(X)
        return predict_proba(model, arch, features, device)

    return predict_fn


def sklearn_predict_fn(model, preprocessor) -> PredictFn:
    """predict_fn for any scikit-learn-API model (.predict_proba)."""

    def predict_fn(engineered_df: pd.DataFrame) -> np.ndarray:
        X = preprocessor.transform_engineered(engineered_df)
        return model.predict_proba(X)[:, 1]

    return predict_fn


def permutation_importance(
    predict_fn: PredictFn, preprocessor, val_df: pd.DataFrame, n_repeats: int = 15, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    engineered = preprocessor.engineer(val_df)
    y_true = engineered[TARGET].to_numpy()

    baseline_probs = predict_fn(engineered)
    baseline_acc = float(((baseline_probs > 0.5).astype(int) == y_true).mean())

    importances: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        drops = []
        for _ in range(n_repeats):
            shuffled = engineered.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            probs = predict_fn(shuffled)
            acc = ((probs > 0.5).astype(int) == y_true).mean()
            drops.append(baseline_acc - acc)
        importances[col] = float(np.mean(drops))

    ranked = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    return {"baseline_accuracy": baseline_acc, "importances": ranked}
