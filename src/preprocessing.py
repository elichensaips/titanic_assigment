"""
Shared preprocessing for the Titanic survival classifier.

Two preprocessor classes share the same engineered features:

- `TitanicPreprocessor` — median/most-frequent impute + scale/one-hot into a
  single flat feature matrix, for the MLP architectures.
- `TitanicTokenizer` — the same features kept as separate per-column tokens
  (scaled numerics, ordinal-encoded categoricals) for the TabTransformer
  architecture, which embeds each column individually.

Both are fit only on the training split and pickled alongside the model
weights so `ds_app.py` can transform new data identically at inference time.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Columns used as model input. PassengerId/Name/Ticket/Cabin are dropped
# (Cabin is >75% missing; Name/Ticket are high-cardinality identifiers) but
# Name is mined for "Title"/surname and Cabin for "Deck"/"HasCabin" below.
NUMERIC_FEATURES = [
    "Age", "Fare", "SibSp", "Parch", "FamilySize",
    "FarePerPerson", "TicketGroupSize", "GroupSurvivalRate",
]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "HasCabin", "Deck", "IsAlone"]
TARGET = "Survived"

_TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}

# Deck "T" has exactly one passenger in the whole dataset — a near-unique
# one-hot column that's pure noise for the model and can easily be entirely
# absent from either the train or validation fold. That one cabin (T) was a
# First Class cabin on the same deck level as the A deck cabins, so folding
# it into "A" is the standard fix (a fixed domain mapping, not something fit
# from data, so it's the same on train/val/inference — no leakage).
_DECK_MAP = {"T": "A"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Title, FamilySize, HasCabin, Deck, TicketGroupSize,
    FarePerPerson, IsAlone from the raw Titanic columns.

    (GroupSurvivalRate and Age imputation are *not* done here — they need
    fitted train-only statistics, so each preprocessor class computes them
    separately via `_GroupSurvivalEncoder` / `_TitleAgeImputer`.)
    """
    df = df.copy()

    df["Title"] = (
        df["Name"].str.extract(r",\s*([^\.]*)\.", expand=False).str.strip()
    )
    df["Title"] = df["Title"].replace(_TITLE_MAP)
    df["Title"] = df["Title"].where(
        df["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Rare"
    )

    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).map({True: "Yes", False: "No"})
    df["HasCabin"] = df["Cabin"].notna().map({True: "Yes", False: "No"})
    df["Deck"] = df["Cabin"].str[0].fillna("U").replace(_DECK_MAP)

    # Recover Deck for a passenger with no recorded Cabin if a ticket-mate
    # (same Ticket number) *does* have one — people sharing a ticket
    # typically shared a cabin. This only ever looks at Ticket/Cabin within
    # whatever rows are being processed (never the target), so it's the
    # same operation at train, val, and inference time — no leakage, just
    # like TicketGroupSize/FarePerPerson below. Doesn't touch HasCabin,
    # which still reflects whether *this* passenger's own cabin was recorded.
    known_deck_by_ticket = df.loc[df["Deck"] != "U"].groupby("Ticket")["Deck"].first()
    recovered_deck = df["Ticket"].map(known_deck_by_ticket)
    df["Deck"] = df["Deck"].where(df["Deck"] != "U", recovered_deck).fillna("U")

    # People traveling on the same ticket number are a travel party (family,
    # servants, friends) — the raw fare on a ticket is the party's *total*
    # fare, so dividing by party size gives a per-person price that's
    # comparable across parties of different sizes.
    ticket_group_size = df.groupby("Ticket")["Ticket"].transform("count")
    df["TicketGroupSize"] = ticket_group_size
    df["FarePerPerson"] = df["Fare"] / ticket_group_size.replace(0, 1)

    return df


class _TitleAgeImputer:
    """Fills missing Age (~20% of rows) using the median age for the
    passenger's Title, rather than one dataset-wide median.

    Age varies hugely by Title — e.g. `Master` (young boys) has a true
    median age of ~3.5 vs. ~28 for the dataset as a whole — so a single
    global median badly mis-imputes exactly the rows where Title already
    tells us a lot about age. Fit on the training split only; unseen/NaN
    titles at transform time fall back to the training set's global median.
    """

    def __init__(self) -> None:
        self._medians: dict[str, float] = {}
        self._global_median: float = 28.0
        self._is_fit = False

    def fit(self, df: pd.DataFrame) -> None:
        self._global_median = float(df["Age"].median())
        self._medians = df.groupby("Title")["Age"].median().to_dict()
        self._is_fit = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("_TitleAgeImputer must be fit before calling transform().")
        df = df.copy()
        fill_values = df["Title"].map(self._medians).astype(float)
        fill_values = fill_values.fillna(self._global_median)
        df["Age"] = df["Age"].fillna(fill_values)
        return df


class _GroupSurvivalEncoder:
    """Leakage-safe "did your travel party survive" feature.

    Widely used in top Titanic solutions: people traveling together
    (same surname + ticket number) tend to share the same fate. Fit on the
    training split only:
      - at fit time, each training row gets its *leave-one-out* group rate
        (excluding its own label) so a row can't just see its own outcome
        reflected back at it;
      - at transform time (validation/inference), a row looks up its group's
        full train-derived rate, or a global fallback if the group was never
        seen in training.
    """

    def __init__(self) -> None:
        self._stats: dict[str, tuple[float, int]] = {}
        self._global_rate: float = 0.5
        self._is_fit = False

    @staticmethod
    def _group_key(df: pd.DataFrame) -> pd.Series:
        surname = df["Name"].str.split(",").str[0].str.strip()
        return surname + "_" + df["Ticket"].astype(str)

    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> np.ndarray:
        key = self._group_key(df)
        grouped = pd.DataFrame({"key": key.to_numpy(), "y": target.to_numpy()}).groupby("key")["y"]
        stats = grouped.agg(["sum", "count"])
        self._stats = {k: (float(r["sum"]), int(r["count"])) for k, r in stats.iterrows()}
        self._global_rate = float(target.mean())
        self._is_fit = True

        totals = key.map(lambda k: self._stats[k][0]).to_numpy(dtype=np.float64)
        counts = key.map(lambda k: self._stats[k][1]).to_numpy(dtype=np.float64)
        y = target.to_numpy(dtype=np.float64)
        denom = counts - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            loo = (totals - y) / denom
        loo = np.where(denom > 0, loo, self._global_rate)
        return loo.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("_GroupSurvivalEncoder must be fit before calling transform().")
        key = self._group_key(df)
        rates = key.map(
            lambda k: (self._stats[k][0] / self._stats[k][1]) if k in self._stats else self._global_rate
        )
        return rates.to_numpy(dtype=np.float32)


class TitanicPreprocessor:
    """Fits a ColumnTransformer (impute+scale numeric, impute+one-hot
    categorical) on the training split and reuses it at inference. Used by
    the `mlp` / `deep_mlp` architectures, which take one flat feature vector.
    """

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
        self._group_encoder = _GroupSurvivalEncoder()
        self._age_imputer = _TitleAgeImputer()
        self._is_fit = False
        self.n_features_: int | None = None

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if TARGET not in df.columns:
            raise RuntimeError("fit_transform requires a 'Survived' column.")
        df = engineer_features(df)
        self._age_imputer.fit(df)
        df = self._age_imputer.transform(df)
        df["GroupSurvivalRate"] = self._group_encoder.fit_transform(df, df[TARGET])
        X = self.column_transformer.fit_transform(
            df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        y = df[TARGET].to_numpy(dtype=np.float32)
        self._is_fit = True
        self.n_features_ = X.shape[1]
        return X.astype(np.float32), y

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineered dataframe (incl. fitted Age imputation and
        GroupSurvivalRate), without the final impute/scale/one-hot step.
        Exposed so feature-importance analysis can permute one column at a
        time before that final step."""
        if not self._is_fit:
            raise RuntimeError("Preprocessor must be fit before calling engineer().")
        df = engineer_features(df)
        df = self._age_imputer.transform(df)
        df["GroupSurvivalRate"] = self._group_encoder.transform(df)
        return df

    def transform_engineered(self, df: pd.DataFrame) -> np.ndarray:
        """Impute/scale/one-hot an already-`engineer()`-ed dataframe."""
        X = self.column_transformer.transform(
            df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        return X.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        if not self._is_fit:
            raise RuntimeError("Preprocessor must be fit before calling transform().")
        edf = self.engineer(df)
        X = self.transform_engineered(edf)
        y = (
            edf[TARGET].to_numpy(dtype=np.float32)
            if TARGET in edf.columns and edf[TARGET].notna().all()
            else None
        )
        return X, y

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TitanicPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


class TitanicTokenizer:
    """Same engineered features as `TitanicPreprocessor`, but kept as
    separate per-column tokens instead of one flat one-hot vector: scaled
    numeric columns plus ordinal-encoded categorical columns (index 0
    reserved for missing/unseen categories). Used by the `tab_transformer`
    architecture, which embeds each column independently before attending
    over them.
    """

    def __init__(self) -> None:
        self._numeric_imputer = SimpleImputer(strategy="median")
        self._numeric_scaler = StandardScaler()
        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        self._cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._group_encoder = _GroupSurvivalEncoder()
        self._age_imputer = _TitleAgeImputer()
        self._is_fit = False
        self.cardinalities: list[int] = []

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if TARGET not in df.columns:
            raise RuntimeError("fit_transform requires a 'Survived' column.")
        df = engineer_features(df)
        self._age_imputer.fit(df)
        df = self._age_imputer.transform(df)
        df["GroupSurvivalRate"] = self._group_encoder.fit_transform(df, df[TARGET])

        num = self._numeric_imputer.fit_transform(df[NUMERIC_FEATURES])
        num = self._numeric_scaler.fit_transform(num).astype(np.float32)

        cat_raw = self._cat_imputer.fit_transform(df[CATEGORICAL_FEATURES])
        cat = self._cat_encoder.fit_transform(cat_raw)
        cat = (cat + 1).astype(np.int64)  # shift so 0 == unknown/missing
        self.cardinalities = [len(c) + 1 for c in self._cat_encoder.categories_]

        self._is_fit = True
        y = df[TARGET].to_numpy(dtype=np.float32)
        return num, cat, y

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineered dataframe (incl. fitted Age imputation and
        GroupSurvivalRate), without the final impute/scale/ordinal-encode
        step. Exposed so feature-importance analysis can permute one column
        at a time first."""
        if not self._is_fit:
            raise RuntimeError("Tokenizer must be fit before calling engineer().")
        df = engineer_features(df)
        df = self._age_imputer.transform(df)
        df["GroupSurvivalRate"] = self._group_encoder.transform(df)
        return df

    def transform_engineered(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Impute/scale/ordinal-encode an already-`engineer()`-ed dataframe."""
        num = self._numeric_imputer.transform(df[NUMERIC_FEATURES])
        num = self._numeric_scaler.transform(num).astype(np.float32)

        cat_raw = self._cat_imputer.transform(df[CATEGORICAL_FEATURES])
        cat = self._cat_encoder.transform(cat_raw)
        cat = (cat + 1).astype(np.int64)
        return num, cat

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if not self._is_fit:
            raise RuntimeError("Tokenizer must be fit before calling transform().")
        edf = self.engineer(df)
        num, cat = self.transform_engineered(edf)
        y = (
            edf[TARGET].to_numpy(dtype=np.float32)
            if TARGET in edf.columns and edf[TARGET].notna().all()
            else None
        )
        return num, cat, y

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TitanicTokenizer":
        with open(path, "rb") as f:
            return pickle.load(f)
