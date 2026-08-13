"""
Fetches train.csv for the Titanic competition directly from Kaggle.

Requirements:
    pip install kaggle
    Kaggle API credentials at ~/.kaggle/kaggle.json
    (Account -> Settings -> API -> "Create New Token" on kaggle.com)
    You must also have accepted the competition rules at:
    https://www.kaggle.com/competitions/titanic/rules

Usage:
    python data/fetch_data.py
"""

import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
COMPETITION = "titanic"


def fetch_via_kaggle_api() -> Path:
    """Download train.csv using the official Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "The 'kaggle' package is not installed. Run: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY env vars)

    print(f"Downloading '{COMPETITION}' competition files to {DATA_DIR} ...")
    api.competition_download_file(COMPETITION, "train.csv", path=str(DATA_DIR))

    zip_path = DATA_DIR / "train.csv.zip"
    csv_path = DATA_DIR / "train.csv"

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    if not csv_path.exists():
        raise SystemExit("Download finished but train.csv was not found.")

    print(f"Saved: {csv_path}")
    return csv_path


if __name__ == "__main__":
    try:
        fetch_via_kaggle_api()
    except SystemExit as e:
        print(f"\n[!] Could not fetch from Kaggle automatically: {e}")
        print(
            "    Manually download 'train.csv' from "
            "https://www.kaggle.com/competitions/titanic/data "
            f"and place it at {DATA_DIR / 'train.csv'}"
        )
        sys.exit(1)
