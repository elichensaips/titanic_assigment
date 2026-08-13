"""
Standalone training script for the Titanic survival classifier.

Loads data/train.csv, creates a train/validation split, fits the
preprocessing pipeline on the training split only, trains and compares
several PyTorch architectures (a small MLP, a deeper MLP, and a small
tabular Transformer), and saves everything needed for inference to
artifacts/:

    artifacts/model.pt              - best-performing model's weights + arch metadata
    artifacts/preprocessor.pkl      - the (matching) fitted preprocessor/tokenizer
    artifacts/val_split.csv         - held-out validation rows
    artifacts/history.json          - per-epoch train/val loss & accuracy, per architecture
    artifacts/model_comparison.json - final val loss/accuracy for every architecture tried
    artifacts/feature_importance.json - permutation feature importance of the winner

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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.architectures import ARCH_NAMES, forward, make_model, make_preprocessor
from src.importance import permutation_importance

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


def run_training(arch, model, train_loader, val_loader, device, args, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
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

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"  [{arch}] epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  [{arch}] early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    return best_state, best_val_loss, best_val_acc, history


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
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["Survived"]
    )
    print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows")
    val_df.to_csv(artifacts_dir / "val_split.csv", index=False)

    results = {}

    for arch in archs:
        print(f"\n=== Training '{arch}' ===")
        set_seed(args.seed)  # identical init/shuffling across architectures for a fair comparison

        preprocessor = make_preprocessor(arch)
        if arch == "tab_transformer":
            num_train, cat_train, y_train = preprocessor.fit_transform(train_df)
            num_val, cat_val, y_val = preprocessor.transform(val_df)
            meta = {"n_numeric": num_train.shape[1], "cat_cardinalities": preprocessor.cardinalities}
            train_feats, val_feats = (num_train, cat_train), (num_val, cat_val)
        else:
            X_train, y_train = preprocessor.fit_transform(train_df)
            X_val, y_val = preprocessor.transform(val_df)
            meta = {"n_features": X_train.shape[1]}
            train_feats, val_feats = X_train, X_val

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
        }
        print(f"  [{arch}] best val_loss={best_val_loss:.4f} val_acc={best_val_acc:.3f}")

    # ---- Pick winner (lowest validation loss) ------------------------------
    winner = min(results, key=lambda a: results[a]["best_val_loss"])
    print(f"\nWinning architecture: '{winner}' (val_loss={results[winner]['best_val_loss']:.4f}, "
          f"val_acc={results[winner]['best_val_acc']:.3f})")

    winner_model = make_model(winner, results[winner]["meta"]).to(device)
    winner_model.load_state_dict(results[winner]["best_state"])

    torch.save(
        {
            "state_dict": results[winner]["best_state"],
            "arch": winner,
            "arch_kwargs": results[winner]["meta"],
        },
        artifacts_dir / "model.pt",
    )
    results[winner]["preprocessor"].save(artifacts_dir / "preprocessor.pkl")

    history_out = {a: results[a]["history"] for a in results}
    history_out["winner"] = winner
    with open(artifacts_dir / "history.json", "w") as f:
        json.dump(history_out, f, indent=2)

    comparison = {
        a: {"best_val_loss": results[a]["best_val_loss"], "best_val_acc": results[a]["best_val_acc"]}
        for a in results
    }
    comparison["winner"] = winner
    with open(artifacts_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nComputing permutation feature importance for the winning model...")
    importance = permutation_importance(
        winner_model, winner, results[winner]["preprocessor"], val_df, device
    )
    with open(artifacts_dir / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    top5 = list(importance["importances"].items())[:5]
    print("Top 5 features by permutation importance (accuracy drop when shuffled):")
    for name, drop in top5:
        print(f"  {name:20s} {drop:+.4f}")

    print(f"\nSaved model -> {artifacts_dir / 'model.pt'}")
    print(f"Saved preprocessor -> {artifacts_dir / 'preprocessor.pkl'}")
    print(f"Saved comparison -> {artifacts_dir / 'model_comparison.json'}")
    print(f"Saved feature importance -> {artifacts_dir / 'feature_importance.json'}")


if __name__ == "__main__":
    main()
