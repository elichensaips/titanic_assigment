"""
Standalone training script for the Titanic survival classifier.

Loads data/train.csv, creates a train/validation split, fits the
preprocessing pipeline on the training split only, trains a PyTorch MLP,
and saves everything needed for inference to artifacts/:

    artifacts/model.pt          - trained model weights (state_dict)
    artifacts/preprocessor.pkl  - fitted TitanicPreprocessor
    artifacts/val_split.csv     - held-out validation rows (for the Streamlit
                                   evaluation tab, so it scores on data the
                                   model never trained on)
    artifacts/history.json      - per-epoch train/val loss & accuracy

Usage:
    python train.py
    python train.py --data data/train.csv --epochs 100 --val-size 0.2
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

from src.model import TitanicNet
from src.preprocessing import TitanicPreprocessor

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Titanic survival classifier")
    p.add_argument("--data", default="data/train.csv", help="Path to Kaggle train.csv")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # ---- Preprocess (fit ONLY on train, to avoid leakage) ------------------
    preprocessor = TitanicPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    preprocessor.save(artifacts_dir / "preprocessor.pkl")
    print(f"Feature dimension: {X_train.shape[1]}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---- Model / optimizer --------------------------------------------------
    model = TitanicNet(n_features=X_train.shape[1]).to(device)
    # Survived is somewhat imbalanced (~38% positive) -> weight the positive class
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            train_n += len(xb)

        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
                val_n += len(xb)

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
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        # ---- Early stopping ----
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {"state_dict": model.state_dict(), "n_features": X_train.shape[1]},
        artifacts_dir / "model.pt",
    )
    with open(artifacts_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved model -> {artifacts_dir / 'model.pt'}")
    print(f"Saved preprocessor -> {artifacts_dir / 'preprocessor.pkl'}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
