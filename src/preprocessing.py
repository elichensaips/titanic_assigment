"""
Shared preprocessing for the Titanic survival classifier.

The same TitanicPreprocessor is used by train.py (fit + transform on the
training split) and ds_app.py (transform-only at inference time), so the
exact same feature engineering is guaranteed on both sides. The fitted
preprocessor is pickled alongside the model weights.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns used as model input. PassengerId/Name/Ticket/Cabin are dropped
# (Cabin is >75% missing; Name/Ticket are high-cardinality identifiers) but
# Name is first mined for "Title" and Cabin for "HasCabin" below.
NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "HasCabin"]
TARGET = "Survived"

_TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Title, FamilySize, HasCabin from the raw Titanic columns."""
    df = df.copy()

    df["Title"] = (
        df["Name"].str.extract(r",\s*([^\.]*)\.", expand=False).str.strip()
    )
    df["Title"] = df["Title"].replace(_TITLE_MAP)
    df["Title"] = df["Title"].where(
        df["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Rare"
    )

    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["HasCabin"] = df["Cabin"].notna().map({True: "Yes", False: "No"})

    return df


class TitanicPreprocessor:
    """Fits a ColumnTransformer on the training split and reuses it at inference."""

    def __init__(self) -> None:
        numeric_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        self.column_transformer = ColumnTransformer(
            [
                ("num", numeric_pipe, NUMERIC_FEATURES),
                ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ]
        )
        self._is_fit = False
        self.n_features_: int | None = None

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = engineer_features(df)
        X = self.column_transformer.fit_transform(
            df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        y = df[TARGET].to_numpy(dtype=np.float32) if TARGET in df.columns else None
        self._is_fit = True
        self.n_features_ = X.shape[1]
        return X.astype(np.float32), y

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        if not self._is_fit:
            raise RuntimeError("Preprocessor must be fit before calling transform().")
        df = engineer_features(df)
        X = self.column_transformer.transform(
            df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        y = (
            df[TARGET].to_numpy(dtype=np.float32)
            if TARGET in df.columns and df[TARGET].notna().all()
            else None
        )
        return X.astype(np.float32), y

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TitanicPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
