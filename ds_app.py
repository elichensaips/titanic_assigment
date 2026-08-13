"""
Streamlit app for the Titanic survival classifier.

Two things live here:
  1. "Validation results"  - shows how the model trained by train.py performed
     on the held-out validation split (metrics + plots), read straight from
     the artifacts/ folder train.py produced.
  2. "Run inference"       - lets the user point at any CSV with the Titanic
     schema, loads the saved model + preprocessor from disk, runs predictions,
     and (if the CSV has a Survived column) shows evaluation plots for it too.

Run:
    streamlit run ds_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import TitanicNet
from src.preprocessing import TitanicPreprocessor

ARTIFACTS_DIR = Path("artifacts")

st.set_page_config(page_title="Titanic Survival Classifier", layout="wide")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_and_preprocessor(artifacts_dir: Path):
    preprocessor = TitanicPreprocessor.load(artifacts_dir / "preprocessor.pkl")
    checkpoint = torch.load(artifacts_dir / "model.pt", map_location="cpu")
    model = TitanicNet(n_features=checkpoint["n_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, preprocessor


@st.cache_data
def load_history(artifacts_dir: Path):
    path = artifacts_dir / "history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def predict(model: TitanicNet, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.sigmoid(logits).numpy()
    return probs, (probs > 0.5).astype(int)


def show_metrics_and_plots(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    col4.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        fig, ax = plt.subplots(figsize=(4, 4))
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Died", "Survived"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    with plot_col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        try:
            RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
            auc = roc_auc_score(y_true, y_proba)
            ax.set_title(f"ROC Curve (AUC = {auc:.3f})")
        except ValueError:
            ax.set_title("ROC Curve unavailable (single class present)")
        st.pyplot(fig)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.title("🚢 Titanic Survival Classifier")

if not (ARTIFACTS_DIR / "model.pt").exists():
    st.error(
        "No trained model found in `artifacts/`. Run `python train.py` first, "
        "then reload this app."
    )
    st.stop()

model, preprocessor = load_model_and_preprocessor(ARTIFACTS_DIR)
history = load_history(ARTIFACTS_DIR)

tab_val, tab_infer = st.tabs(["📊 Validation results", "🔮 Run inference"])

# ---- Tab 1: results on the held-out split from train.py -------------------
with tab_val:
    st.subheader("Performance on the held-out validation split")
    st.caption(
        "This is the split `train.py` set aside before fitting the model "
        "(saved to `artifacts/val_split.csv`) — the model never trained on these rows."
    )

    val_path = ARTIFACTS_DIR / "val_split.csv"
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        X_val, y_val = preprocessor.transform(val_df)
        probs, preds = predict(model, X_val)
        show_metrics_and_plots(y_val, preds, probs)
    else:
        st.warning("`artifacts/val_split.csv` not found — re-run train.py to generate it.")

    if history:
        st.subheader("Training curves")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.plot(history["train_loss"], label="train")
        ax1.plot(history["val_loss"], label="validation")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax2.plot(history["train_acc"], label="train")
        ax2.plot(history["val_acc"], label="validation")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        st.pyplot(fig)

# ---- Tab 2: user-supplied CSV -----------------------------------------------
with tab_infer:
    st.subheader("Run inference on a CSV")
    st.caption(
        "Provide a CSV with the standard Titanic columns "
        "(PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked). "
        "If it also has a `Survived` column, evaluation metrics/plots are shown."
    )

    csv_path = st.text_input("Path to CSV file", value="data/sample_train.csv")
    uploaded = st.file_uploader("...or upload a CSV", type="csv")

    df_input = None
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
    elif csv_path:
        p = Path(csv_path)
        if p.exists():
            df_input = pd.read_csv(p)
        else:
            st.warning(f"File not found: {csv_path}")

    if df_input is not None:
        st.write(f"Loaded **{len(df_input)}** rows.")
        st.dataframe(df_input.head(10), use_container_width=True)

        try:
            X, y = preprocessor.transform(df_input)
        except Exception as e:
            st.error(f"Preprocessing failed — check the CSV has the expected columns. ({e})")
            st.stop()

        if st.button("Run inference", type="primary"):
            probs, preds = predict(model, X)
            result_df = df_input.copy()
            result_df["PredictedSurvival"] = preds
            result_df["SurvivalProbability"] = probs.round(3)
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download predictions as CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

            if y is not None:
                st.subheader("Evaluation (ground truth found in CSV)")
                show_metrics_and_plots(y, preds, probs)
            else:
                st.info("No usable `Survived` column found — showing predictions only.")
