"""
Streamlit app for the Titanic survival classifier.

Three things live here:
  1. A model picker — train.py trains and saves every PyTorch architecture
     (mlp, deep_mlp, tab_transformer), and the optional, bonus
     train_baselines.py additionally saves classical scikit-learn-API
     baselines (Logistic Regression, KNN, Naive Bayes, SVM, Random Forest,
     HistGradientBoosting, XGBoost, LightGBM, CatBoost). Every model found
     in artifacts/ shows up here so the user can choose which one drives the
     two tabs below. Defaults to tab_transformer (highest CV accuracy of
     every model here). train.py's own pick — mlp, by lowest CV loss among
     just the 3 PyTorch architectures — is labeled separately and still one
     selection away. **The assignment's required deliverable is train.py's
     PyTorch model** (mlp / deep_mlp / tab_transformer, all three shown) —
     the classical baselines are bonus context, clearly labeled as such below.
  2. "Validation results"  - shows how the selected model performed on the
     held-out validation split, the full model comparison table, and
     permutation feature importance — all read straight from artifacts/.
  3. "Run inference"       - lets the user point at any CSV with the Titanic
     schema, loads the selected model + preprocessor from disk, runs
     predictions, and (if the CSV has a Survived column) shows evaluation
     plots for it too.

Run:
    streamlit run ds_app.py
"""

from __future__ import annotations

import json
import pickle
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

from src.architectures import ARCH_NAMES, make_model, predict_proba

ARTIFACTS_DIR = Path("artifacts")
DEVICE = torch.device("cpu")  # small model + tiny batches -> CPU is plenty for the app

st.set_page_config(page_title="Titanic Survival Classifier", layout="wide")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
def is_pytorch(name: str) -> bool:
    return name in ARCH_NAMES


@st.cache_resource
def load_model_and_preprocessor(artifacts_dir: Path, name: str):
    if is_pytorch(name):
        with open(artifacts_dir / f"preprocessor_{name}.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        checkpoint = torch.load(artifacts_dir / f"model_{name}.pt", map_location="cpu")
        model = make_model(name, checkpoint["arch_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, preprocessor
    # Classical baseline: plain pickled sklearn-API estimator + one shared
    # preprocessor (every baseline uses the same flat one-hot feature vector).
    with open(artifacts_dir / f"model_{name}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(artifacts_dir / "preprocessor_baselines.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor


@st.cache_data
def load_json(artifacts_dir: Path, name: str):
    path = artifacts_dir / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def transform_for_inference(preprocessor, name: str, df: pd.DataFrame):
    if name == "tab_transformer":
        num, cat, y = preprocessor.transform(df)
        return (torch.from_numpy(num), torch.from_numpy(cat)), y
    X, y = preprocessor.transform(df)
    return (X if not is_pytorch(name) else torch.from_numpy(X)), y


def predict(model, name: str, features) -> tuple[np.ndarray, np.ndarray]:
    if is_pytorch(name):
        probs = predict_proba(model, name, features, DEVICE)
    else:
        probs = model.predict_proba(features)[:, 1]
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

pytorch_comparison = load_json(ARTIFACTS_DIR, "model_comparison.json") or {}
baseline_comparison = load_json(ARTIFACTS_DIR, "baseline_comparison.json") or {}
comparison = {
    **{k: v for k, v in pytorch_comparison.items() if isinstance(v, dict)},
    **{k: v for k, v in baseline_comparison.items() if isinstance(v, dict)},
}
cv_folds = pytorch_comparison.get("cv_folds") or baseline_comparison.get("cv_folds")

if not comparison:
    st.error(
        "No trained models found in `artifacts/`. Run `python train.py` first "
        "(and optionally `python train_baselines.py` for the classical baselines), "
        "then reload this app."
    )
    st.stop()

available_names = sorted(comparison, key=lambda n: comparison[n]["cv_mean_val_loss"])
# train.py's own winner (by lowest CV loss among the 3 required PyTorch
# architectures) is still labeled and always visible, but the app's default
# selection is pinned to tab_transformer — it has the highest CV accuracy of
# every model here, PyTorch or baseline (see README "Example usage" for the
# full loss-vs-accuracy discussion behind train.py's own, different pick).
pytorch_winner = pytorch_comparison.get("winner")
overall_best = min(comparison, key=lambda n: comparison[n]["cv_mean_val_loss"])
default_selection = "tab_transformer" if "tab_transformer" in comparison else (
    pytorch_winner if pytorch_winner in comparison else overall_best
)


def _label(name: str) -> str:
    cv_acc = comparison[name]["cv_mean_val_acc"]
    kind = "PyTorch" if is_pytorch(name) else "classical baseline"
    tags = []
    if name == default_selection:
        tags.append("📌 loads by default — highest CV accuracy")
    if name == pytorch_winner and name != default_selection:
        tags.append("⭐ train.py's own CV-loss winner")
    if name == overall_best and name not in (default_selection, pytorch_winner):
        tags.append("🏆 lowest CV loss overall")
    suffix = f" ({', '.join(tags)})" if tags else ""
    return f"{name}  —  {kind}  —  {cv_acc:.1%} CV accuracy{suffix}"


arch = st.selectbox(
    "Model",
    options=available_names,
    index=available_names.index(default_selection),
    format_func=_label,
    help="train.py's mlp / deep_mlp / tab_transformer are the assignment's required "
    "PyTorch deliverable. 📌 tab_transformer is pre-selected by default (highest CV "
    "accuracy overall); ⭐ marks train.py's own pick instead, chosen by lowest CV loss "
    "among just the 3 PyTorch architectures; 🏆 marks the lowest CV loss across *all* "
    "models, PyTorch or not. Any name that isn't mlp/deep_mlp/tab_transformer comes "
    "from the optional, bonus train_baselines.py script.",
)
if not is_pytorch(arch):
    st.info(
        f"**{arch}** is a classical scikit-learn-API baseline from the bonus "
        "`train_baselines.py` script, shown for comparison — the assignment's required "
        "trained/saved model is always one of train.py's PyTorch architectures "
        f"({', '.join(ARCH_NAMES)})."
    )

model, preprocessor = load_model_and_preprocessor(ARTIFACTS_DIR, arch)
history = load_json(ARTIFACTS_DIR, "history.json")
importance_all = {
    **(load_json(ARTIFACTS_DIR, "feature_importance.json") or {}),
    **(load_json(ARTIFACTS_DIR, "baseline_feature_importance.json") or {}),
}
importance = importance_all.get(arch)

tab_val, tab_infer = st.tabs(["📊 Validation results", "🔮 Run inference"])

# ---- Tab 1: results on the held-out split from train.py -------------------
with tab_val:
    st.subheader("Performance on the held-out validation split")
    st.caption(
        f"Showing **{arch}**. This is the held-out split `train.py` set aside before "
        "fitting any model (saved to `artifacts/val_split.csv`) — none of the "
        "models trained on these rows."
    )

    val_path = ARTIFACTS_DIR / "val_split.csv"
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        features, y_val = transform_for_inference(preprocessor, arch, val_df)
        probs, preds = predict(model, arch, features)
        show_metrics_and_plots(y_val, preds, probs)
    else:
        st.warning("`artifacts/val_split.csv` not found — re-run train.py to generate it.")

    st.subheader("Model comparison")
    st.caption(
        (
            f"Ranked by lowest mean {cv_folds}-fold CV validation loss on the training split "
            "(more robust than the single held-out split, which is small enough that close "
            "models can flip rank from noise alone). 'kind' distinguishes the required PyTorch "
            "models from the bonus classical baselines. Note: CV loss and CV accuracy don't "
            "always agree — KNN and Naive Bayes tend to output poorly-calibrated probabilities "
            "(confidently wrong more often than well-calibrated models), so they can rank much "
            "worse by loss than their accuracy alone would suggest."
        )
        if cv_folds
        else "Ranked by lowest validation loss."
    )
    comp_df = pd.DataFrame(
        [
            {
                "model": a,
                "kind": "PyTorch" if is_pytorch(a) else "classical baseline",
                "cv_val_loss": v.get("cv_mean_val_loss"),
                "cv_val_loss_std": v.get("cv_std_val_loss"),
                "cv_val_accuracy": v.get("cv_mean_val_acc"),
                "held_out_val_loss": v["best_val_loss"],
                "held_out_val_accuracy": v["best_val_acc"],
            }
            for a, v in comparison.items()
        ]
    ).sort_values("cv_val_loss")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if importance:
        st.subheader(f"Feature importance (permutation, {arch})")
        st.caption(
            "Each feature is shuffled in the validation set and the resulting drop in "
            "accuracy is measured — a larger drop means the model relies on that feature more. "
            f"Baseline validation accuracy: {importance['baseline_accuracy']:.3f}."
        )
        imp_series = pd.Series(importance["importances"]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(imp_series.index, imp_series.values, color="#4C78A8")
        ax.set_xlabel("Accuracy drop when shuffled")
        ax.set_title("Permutation feature importance")
        st.pyplot(fig)

    if history and arch in history:
        st.subheader("Training curves")
        arch_history = history[arch]
        total_epochs = len(arch_history["train_loss"])
        best_epoch = arch_history.get("best_epoch") or total_epochs

        st.caption(
            f"Full training run ({total_epochs} epochs) — the dashed line marks epoch "
            f"{best_epoch}, the lowest validation loss, i.e. the weights actually saved to "
            f"`model_{arch}.pt`. Training continued past it (early-stopping patience) without "
            "improving further, so the curve after the line shows the model overfitting past "
            "the point it was checkpointed at — that tail was never saved."
            if best_epoch < total_epochs
            else f"Trained for all {total_epochs} epochs without early stopping triggering."
        )

        epochs = list(range(1, total_epochs + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.plot(epochs, arch_history["train_loss"], label="train")
        ax1.plot(epochs, arch_history["val_loss"], label="validation")
        ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best epoch (saved)")
        ax1.set_title(f"Loss ({arch})")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax2.plot(epochs, arch_history["train_acc"], label="train")
        ax2.plot(epochs, arch_history["val_acc"], label="validation")
        ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best epoch (saved)")
        ax2.set_title(f"Accuracy ({arch})")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        st.pyplot(fig)
    elif not is_pytorch(arch):
        st.caption("(No per-epoch training curve for classical baselines — they aren't trained iteratively like a neural net.)")

# ---- Tab 2: user-supplied CSV -----------------------------------------------
with tab_infer:
    st.subheader("Run inference on a CSV")
    st.caption(
        f"Using **{arch}** (change the model picker above to switch). Provide a CSV with "
        "the standard Titanic columns (PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, "
        "Ticket, Fare, Cabin, Embarked). If it also has a `Survived` column, evaluation "
        "metrics/plots are shown."
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
            features, y = transform_for_inference(preprocessor, arch, df_input)
        except Exception as e:
            st.error(f"Preprocessing failed — check the CSV has the expected columns. ({e})")
            st.stop()

        if st.button("Run inference", type="primary"):
            probs, preds = predict(model, arch, features)
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
