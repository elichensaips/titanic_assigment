"""
Standalone training script for the Titanic survival classifier.

Loads data/train.csv, creates a train/validation split, fits the
preprocessing pipeline on the training split only, trains and compares
several PyTorch architectures (a small MLP, a deeper MLP, and a small
tabular Transformer), and saves everything needed for inference to
artifacts/.

Architecture selection uses stratified K-fold cross-validation on the
training split (not the single held-out val split) — with only ~700 rows,
a single 80/20 split is noisy enough that two close architectures can flip
rank from one split to another. The held-out val split is still what gets
reported/saved (that's the assignment's required train/validation
evaluation split), CV is just a more robust tie-breaker for *which*
architecture to recommend as the default.

Every trained architecture is saved (not just the winner), so ds_app.py can
let the user pick which model to run:

    artifacts/model_<arch>.pt         - each architecture's weights + metadata
    artifacts/preprocessor_<arch>.pkl - each architecture's fitted preprocessor/tokenizer
    artifacts/val_split.csv           - held-out validation rows (shared, same split for all)
    artifacts/history.json            - per-epoch train/val loss & accuracy, per architecture
    artifacts/model_comparison.json   - CV + held-out val loss/accuracy per architecture, and the winner
    artifacts/feature_importance.json - permutation feature importance, per architecture

Usage:
    python train.py
    python train.py --data data/train.csv --epochs 100 --val-size 0.2
    python train.py --archs mlp,deep_mlp   # skip the transformer
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.architectures import ARCH_NAMES, forward, make_model, make_preprocessor
from src.data import load_train_csv
from src.importance import permutation_importance, torch_predict_fn

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compare Titanic survival classifiers")
    p.add_argument("--data", default="data/train.csv", help="Path to Kaggle train.csv")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--archs",
        default=",".join(ARCH_NAMES),
        help=f"Comma-separated architectures to train and compare, from {ARCH_NAMES}",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified K-fold CV on the training split, used to pick the winning architecture",
    )
    return p.parse_args()


def build_loaders(arch: str, train_feats, train_y, val_feats, val_y, batch_size: int):
    if arch == "tab_transformer":
        num_train, cat_train = train_feats
        num_val, cat_val = val_feats
        train_ds = TensorDataset(
            torch.from_numpy(num_train), torch.from_numpy(cat_train), torch.from_numpy(train_y)
        )
        val_ds = TensorDataset(
            torch.from_numpy(num_val), torch.from_numpy(cat_val), torch.from_numpy(val_y)
        )
    else:
        train_ds = TensorDataset(torch.from_numpy(train_feats), torch.from_numpy(train_y))
        val_ds = TensorDataset(torch.from_numpy(val_feats), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def unpack_batch(arch: str, batch, device):
    if arch == "tab_transformer":
        num_b, cat_b, yb = batch
        return (num_b.to(device), cat_b.to(device)), yb.to(device)
    xb, yb = batch
    return xb.to(device), yb.to(device)


def run_training(arch, model, train_loader, val_loader, device, args, pos_weight, verbose=True):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0
        for batch in train_loader:
            features, yb = unpack_batch(arch, batch, device)
            optimizer.zero_grad()
            logits = forward(model, arch, features)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            train_correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            train_n += len(yb)

        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                features, yb = unpack_batch(arch, batch, device)
                logits = forward(model, arch, features)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(yb)
                val_correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
                val_n += len(yb)

        train_loss /= train_n
        val_loss /= val_n
        train_acc = train_correct / train_n
        val_acc = val_correct / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == args.epochs):
            print(
                f"  [{arch}] epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                if verbose:
                    print(f"  [{arch}] early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    history["best_epoch"] = best_epoch
    return best_state, best_val_loss, best_val_acc, history


def make_features(arch, preprocessor, df, fit: bool):
    """Fit or transform `df` with the right preprocessor call for `arch`,
    returning (features, y, meta) where `features` is an ndarray for
    mlp/deep_mlp or an (x_num, x_cat) ndarray pair for tab_transformer."""
    if arch == "tab_transformer":
        if fit:
            num, cat, y = preprocessor.fit_transform(df)
            meta = {"n_numeric": num.shape[1], "cat_cardinalities": preprocessor.cardinalities}
        else:
            num, cat, y = preprocessor.transform(df)
            meta = None
        return (num, cat), y, meta
    if fit:
        X, y = preprocessor.fit_transform(df)
        meta = {"n_features": X.shape[1]}
    else:
        X, y = preprocessor.transform(df)
        meta = None
    return X, y, meta


def cross_validate(arch: str, train_df: pd.DataFrame, args, device) -> dict:
    """Stratified K-fold CV on the training split only (never touches
    val_df), used purely to pick a winning architecture more robustly than
    a single noisy 80/20 split would."""
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_losses, fold_accs = [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df["Survived"]), start=1):
        set_seed(args.seed + fold_idx)
        fold_train = train_df.iloc[tr_idx]
        fold_val = train_df.iloc[va_idx]

        preprocessor = make_preprocessor(arch)
        train_feats, y_train, meta = make_features(arch, preprocessor, fold_train, fit=True)
        val_feats, y_val, _ = make_features(arch, preprocessor, fold_val, fit=False)

        train_loader, val_loader = build_loaders(arch, train_feats, y_train, val_feats, y_val, args.batch_size)
        model = make_model(arch, meta).to(device)
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
        ).to(device)

        _, fold_val_loss, fold_val_acc, _ = run_training(
            arch, model, train_loader, val_loader, device, args, pos_weight, verbose=False
        )
        fold_losses.append(fold_val_loss)
        fold_accs.append(fold_val_acc)
        print(f"  [{arch}] fold {fold_idx}/{args.cv_folds}: val_loss={fold_val_loss:.4f} val_acc={fold_val_acc:.3f}")

    return {
        "cv_mean_val_loss": float(np.mean(fold_losses)),
        "cv_std_val_loss": float(np.std(fold_losses)),
        "cv_mean_val_acc": float(np.mean(fold_accs)),
        "cv_std_val_acc": float(np.std(fold_accs)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    for a in archs:
        if a not in ARCH_NAMES:
            raise SystemExit(f"Unknown architecture '{a}'. Choose from {ARCH_NAMES}")
    print(f"Device: {device}")
    print(f"Architectures to train: {archs}")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & split -----------------------------------------------------
    df = load_train_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["Survived"]
    )
    print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows")
    val_df.to_csv(artifacts_dir / "val_split.csv", index=False)

    results = {}

    for arch in archs:
        print(f"\n=== Architecture '{arch}' ===")
        print(f"  Running {args.cv_folds}-fold stratified CV on the training split...")
        cv_stats = cross_validate(arch, train_df, args, device)
        print(
            f"  [{arch}] CV: val_loss={cv_stats['cv_mean_val_loss']:.4f}+/-{cv_stats['cv_std_val_loss']:.4f} "
            f"val_acc={cv_stats['cv_mean_val_acc']:.3f}+/-{cv_stats['cv_std_val_acc']:.3f}"
        )

        print(f"  Training on the full train/val split for final artifacts...")
        set_seed(args.seed)  # identical init/shuffling across architectures for a fair comparison

        preprocessor = make_preprocessor(arch)
        train_feats, y_train, meta = make_features(arch, preprocessor, train_df, fit=True)
        val_feats, y_val, _ = make_features(arch, preprocessor, val_df, fit=False)

        train_loader, val_loader = build_loaders(arch, train_feats, y_train, val_feats, y_val, args.batch_size)

        model = make_model(arch, meta).to(device)
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
        ).to(device)

        best_state, best_val_loss, best_val_acc, history = run_training(
            arch, model, train_loader, val_loader, device, args, pos_weight
        )

        results[arch] = {
            "best_state": best_state,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "history": history,
            "meta": meta,
            "preprocessor": preprocessor,
            **cv_stats,
        }
        print(f"  [{arch}] held-out split: val_loss={best_val_loss:.4f} val_acc={best_val_acc:.3f}")

    # ---- Pick winner by CV mean val loss (more robust than the single,
    # noisy 80/20 held-out split) ------------------------------------------
    winner = min(results, key=lambda a: results[a]["cv_mean_val_loss"])
    print(
        f"\nWinning architecture (lowest {args.cv_folds}-fold CV val loss): '{winner}' "
        f"(cv_val_loss={results[winner]['cv_mean_val_loss']:.4f}+/-{results[winner]['cv_std_val_loss']:.4f}, "
        f"held-out val_acc={results[winner]['best_val_acc']:.3f})"
    )

    # ---- Save every trained architecture (not just the winner), so the
    # Streamlit app can let the user pick which one to run --------------------
    history_out = {a: results[a]["history"] for a in results}
    history_out["winner"] = winner
    with open(artifacts_dir / "history.json", "w") as f:
        json.dump(history_out, f, indent=2)

    comparison = {
        a: {
            "best_val_loss": results[a]["best_val_loss"],
            "best_val_acc": results[a]["best_val_acc"],
            "cv_mean_val_loss": results[a]["cv_mean_val_loss"],
            "cv_std_val_loss": results[a]["cv_std_val_loss"],
            "cv_mean_val_acc": results[a]["cv_mean_val_acc"],
            "cv_std_val_acc": results[a]["cv_std_val_acc"],
        }
        for a in results
    }
    comparison["winner"] = winner
    comparison["cv_folds"] = args.cv_folds
    with open(artifacts_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nSaving all trained architectures and their permutation feature importance...")
    importance_out = {}
    for a in results:
        model = make_model(a, results[a]["meta"]).to(device)
        model.load_state_dict(results[a]["best_state"])

        torch.save(
            {"state_dict": results[a]["best_state"], "arch": a, "arch_kwargs": results[a]["meta"]},
            artifacts_dir / f"model_{a}.pt",
        )
        results[a]["preprocessor"].save(artifacts_dir / f"preprocessor_{a}.pkl")

        predict_fn = torch_predict_fn(model, a, results[a]["preprocessor"], device)
        importance_out[a] = permutation_importance(predict_fn, results[a]["preprocessor"], val_df)
        print(f"  [{a}] saved model_{a}.pt, preprocessor_{a}.pkl" + (" (winner)" if a == winner else ""))

    importance_out["winner"] = winner
    with open(artifacts_dir / "feature_importance.json", "w") as f:
        json.dump(importance_out, f, indent=2)

    top5 = list(importance_out[winner]["importances"].items())[:5]
    print(f"\nTop 5 features by permutation importance for the winner ('{winner}'):")
    for name, drop in top5:
        print(f"  {name:20s} {drop:+.4f}")

    print(f"\nSaved {len(results)} model(s) + preprocessor(s) to {artifacts_dir}/")
    print(f"Saved comparison -> {artifacts_dir / 'model_comparison.json'}")
    print(f"Saved feature importance -> {artifacts_dir / 'feature_importance.json'}")


if __name__ == "__main__":
    main()
