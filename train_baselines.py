"""
Trains classical scikit-learn-API baseline models on the Titanic dataset.

This is a **bonus, non-graded script** — the assignment's required
deliverable is train.py's PyTorch model. This exists purely so ds_app.py's
model picker can also show how well-known classical baselines do, right
next to mlp / deep_mlp / tab_transformer, using the identical train/val
split, feature engineering, and 5-fold CV methodology as train.py (see
notebooks/model_benchmark.ipynb for the same comparison with plots/discussion
— this script's job is just to persist the models as artifacts for the app).

Trains: Logistic Regression, KNN, Naive Bayes, SVM, Random Forest,
HistGradientBoosting, and (if installed) XGBoost, LightGBM, CatBoost.

Saves to artifacts/:
    model_<name>.pkl                 - each baseline's fitted estimator
    preprocessor_baselines.pkl       - shared TitanicPreprocessor (every
                                        baseline uses the same flat one-hot
                                        feature vector)
    baseline_comparison.json         - CV + held-out val loss/accuracy per baseline
    baseline_feature_importance.json - permutation feature importance per baseline

Usage:
    python train_baselines.py
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.importance import permutation_importance, sklearn_predict_fn
from src.preprocessing import TitanicPreprocessor

SEED = 42


def make_baselines(seed: int) -> dict:
    baselines = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "svm": SVC(probability=True, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        baselines["xgboost"] = XGBClassifier(random_state=seed, eval_metric="logloss")
    except ImportError:
        print("xgboost not installed - skipping (pip install xgboost to include it)")
    try:
        from lightgbm import LGBMClassifier

        baselines["lightgbm"] = LGBMClassifier(random_state=seed, verbose=-1)
    except ImportError:
        print("lightgbm not installed - skipping (pip install lightgbm to include it)")
    try:
        from catboost import CatBoostClassifier

        baselines["catboost"] = CatBoostClassifier(random_state=seed, verbose=False)
    except ImportError:
        print("catboost not installed - skipping (pip install catboost to include it)")
    return baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train classical baseline models (bonus, non-graded)")
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["Survived"]
    )
    print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows")

    # Same held-out split as train.py (identical seed/val-size), so this
    # reuses the same artifacts/val_split.csv rather than writing its own.
    preprocessor = TitanicPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    preprocessor.save(artifacts_dir / "preprocessor_baselines.pkl")

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    comparison: dict = {}
    importance_out: dict = {}

    for name, clf in make_baselines(args.seed).items():
        print(f"\n=== {name} ===")

        # ---- 5-fold CV on the training split (mirrors train.py's methodology) ----
        fold_accs, fold_losses = [], []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df["Survived"]), start=1):
            fold_train = train_df.iloc[tr_idx]
            fold_val = train_df.iloc[va_idx]
            fold_pre = TitanicPreprocessor()
            X_tr, y_tr = fold_pre.fit_transform(fold_train)
            X_va, y_va = fold_pre.transform(fold_val)

            fold_clf = clone(clf)
            fold_clf.fit(X_tr, y_tr)
            proba = fold_clf.predict_proba(X_va)[:, 1]
            preds = (proba > 0.5).astype(int)
            fold_accs.append(accuracy_score(y_va, preds))
            fold_losses.append(log_loss(y_va, proba, labels=[0, 1]))
            print(f"  fold {fold_idx}/{args.cv_folds}: acc={fold_accs[-1]:.3f} loss={fold_losses[-1]:.4f}")

        cv_mean_acc, cv_std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
        cv_mean_loss, cv_std_loss = float(np.mean(fold_losses)), float(np.std(fold_losses))
        print(f"  CV: acc={cv_mean_acc:.3f}+/-{cv_std_acc:.3f} loss={cv_mean_loss:.4f}+/-{cv_std_loss:.4f}")

        # ---- Fit on the full training split for the final saved model ----
        clf.fit(X_train, y_train)
        proba_val = clf.predict_proba(X_val)[:, 1]
        preds_val = (proba_val > 0.5).astype(int)
        held_out_acc = float(accuracy_score(y_val, preds_val))
        held_out_loss = float(log_loss(y_val, proba_val, labels=[0, 1]))
        print(f"  held-out split: acc={held_out_acc:.3f} loss={held_out_loss:.4f}")

        with open(artifacts_dir / f"model_{name}.pkl", "wb") as f:
            pickle.dump(clf, f)

        comparison[name] = {
            "best_val_loss": held_out_loss,
            "best_val_acc": held_out_acc,
            "cv_mean_val_loss": cv_mean_loss,
            "cv_std_val_loss": cv_std_loss,
            "cv_mean_val_acc": cv_mean_acc,
            "cv_std_val_acc": cv_std_acc,
        }

        predict_fn = sklearn_predict_fn(clf, preprocessor)
        importance_out[name] = permutation_importance(predict_fn, preprocessor, val_df)

    winner = min(comparison, key=lambda n: comparison[n]["cv_mean_val_loss"])
    comparison["winner"] = winner
    comparison["cv_folds"] = args.cv_folds
    with open(artifacts_dir / "baseline_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    importance_out["winner"] = winner
    with open(artifacts_dir / "baseline_feature_importance.json", "w") as f:
        json.dump(importance_out, f, indent=2)

    n_models = len(comparison) - 2  # minus "winner" / "cv_folds" keys
    print(f"\nBest classical baseline (lowest CV loss): '{winner}'")
    print(f"Saved {n_models} baseline model(s) + shared preprocessor to {artifacts_dir}/")
    print(f"Saved comparison -> {artifacts_dir / 'baseline_comparison.json'}")
    print(f"Saved feature importance -> {artifacts_dir / 'baseline_feature_importance.json'}")


if __name__ == "__main__":
    main()
