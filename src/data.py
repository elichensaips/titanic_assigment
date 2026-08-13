"""Shared, friendly data loading for train.py and train_baselines.py."""

from __future__ import annotations

import sys

import pandas as pd


def load_train_csv(path: str) -> pd.DataFrame:
    """Load the Titanic training CSV, with a clear error (not a raw
    traceback) if the file is missing or malformed."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(
            f"\n[!] Could not find '{path}'.\n"
            "    Fetch it first: python data/fetch_data.py\n"
            "    (requires Kaggle API credentials — see README's \"Getting the data\" section)\n"
            "    Or, to try the pipeline without Kaggle credentials:\n"
            "    python train.py --data data/sample_train.csv"
        )
    except pd.errors.ParserError as e:
        sys.exit(f"\n[!] '{path}' isn't a valid CSV: {e}")

    if "Survived" not in df.columns:
        sys.exit(
            f"\n[!] '{path}' has no 'Survived' column — this script needs the labeled "
            "Titanic training data (train.csv), not test.csv."
        )
    return df
