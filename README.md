# Titanic Survival Classifier

Data Science home assignment — end-to-end classification pipeline (EDA →
PyTorch training script → Streamlit evaluation/inference app) on the
[Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data).

## Architecture & design choices

```
.
├── data/
│   ├── fetch_data.py       # pulls train.csv from Kaggle via the official API
│   └── sample_train.csv    # small (50-row) sample committed to the repo
├── notebooks/
│   ├── eda.ipynb            # exploratory data analysis
│   └── model_benchmark.ipynb # bonus: PyTorch archs vs. classic SOTA baselines
├── src/
│   ├── __init__.py          # makes src a regular package (avoids shadowing
│   │                          # by any same-named package on sys.path)
│   ├── preprocessing.py     # TitanicPreprocessor / TitanicTokenizer: shared
│   │                          # feature engineering
│   ├── model.py              # TitanicNet (MLP) + TabTransformerNet
│   ├── architectures.py      # registry: arch name -> preprocessor/model/forward
│   └── importance.py         # permutation feature importance (model-agnostic)
├── train.py                  # standalone training + architecture-comparison script
├── ds_app.py                 # Streamlit app: validation results + inference UI
├── requirements.txt
└── artifacts/                 # created by train.py (gitignored): model.pt,
                                # preprocessor.pkl, val_split.csv, history.json,
                                # model_comparison.json, feature_importance.json
```

**Why a shared preprocessor.** The exact same feature engineering and fitted
scalers/encoders have to be used at training and inference time, or the model
silently sees a different feature distribution than it was trained on.
`src/preprocessing.py` is imported by both `train.py` (fit + transform) and
`ds_app.py` (transform-only, loaded from a pickle), so there is a single
source of truth. It exposes two classes built on the same engineered
features:
- `TitanicPreprocessor` — impute + scale/one-hot into one flat feature
  vector, for the MLP architectures.
- `TitanicTokenizer` — the same features kept as separate per-column tokens
  (scaled numerics, ordinal-encoded categoricals), for the TabTransformer.

**Feature engineering** (see `notebooks/eda.ipynb` for the analysis behind
each choice):
- `Title` extracted from `Name` (Mr/Mrs/Miss/Master/Rare) — captures
  age/gender/class signal `Name` itself can't be used for directly.
- `FamilySize = SibSp + Parch + 1`, `IsAlone` — survival is non-monotonic in
  raw `SibSp`/`Parch` but has a cleaner U-shape in total family size.
- `HasCabin` / `Deck` — a presence flag plus the cabin letter, rather than
  the raw (77%-missing) `Cabin` string.
- `TicketGroupSize` / `FarePerPerson` — people sharing a ticket number are a
  travel party; the raw `Fare` is the *party's* fare, so dividing by party
  size gives a per-person price comparable across parties.
- `GroupSurvivalRate` — the single most informative engineered feature
  (commonly used in top Titanic solutions): passengers traveling together
  (same surname + ticket) tend to share the same fate. Computed
  **leakage-safe**: each training row gets its *leave-one-out* group rate
  (excluding its own label), and validation/inference rows look up their
  group's train-derived rate, falling back to the global training rate for
  unseen groups (`src/preprocessing.py::_GroupSurvivalEncoder`).
- `PassengerId`, `Name`, `Ticket`, and raw `Cabin` are dropped as either
  identifiers or too sparse to use directly (after being mined for the
  features above).
- Everything is **fit only on the training split** to avoid leakage into
  validation.

**Models — three PyTorch architectures, compared via cross-validation.**
`train.py` trains all three:
- `mlp` — small 2-hidden-layer MLP (32→16 units, ReLU, dropout 0.3).
- `deep_mlp` — a deeper variant (64→32→16 units) for more capacity.
- `tab_transformer` — a small FT-Transformer-style network
  (`src/model.py::TabTransformerNet`): each feature (numeric or categorical)
  is embedded into its own token, a learnable `[CLS]` token is prepended,
  and a `nn.TransformerEncoder` attends over them; the `[CLS]` output feeds
  the classification head. Included as a genuine architecture comparison,
  not because it's expected to dominate — attention over ~20 tokens is a lot
  of extra capacity for ~700 training rows, and the benchmark notebook
  discusses that tradeoff.

All three are trained with `BCEWithLogitsLoss` using a `pos_weight` to
correct for the ~62/38 class imbalance, Adam with weight decay, and early
stopping on validation loss.

**Why cross-validation for architecture selection.** With only ~700
training rows, a single 80/20 split is noisy enough that two close
architectures can flip rank depending on which rows happen to land in
validation — early iterations of this project picked a "winner" by a
val-loss margin of 0.004, which is noise, not signal. `train.py` now runs
**5-fold stratified cross-validation on the training split** (`--cv-folds`,
default 5) for each architecture and picks the one with the lowest mean CV
validation loss. The single held-out split (below) is still what's kept
untouched throughout — CV only ever sees the training portion — and is what
gets reported as the final, assignment-required validation metric.

**Evaluation.** `train.py` holds out a stratified validation split
(`--val-size`, default 20%) *before* fitting or cross-validating anything,
and saves it to `artifacts/val_split.csv` so the Streamlit app can score the
trained model on data it never saw. It also saves
`artifacts/model_comparison.json` (both the CV mean±std and the held-out
split's loss/accuracy, for every architecture tried) and
`artifacts/feature_importance.json` (permutation importance of the winner —
see below). The app reports accuracy/precision/recall/F1, a confusion
matrix, an ROC curve, the architecture comparison table, a feature
importance chart, and training curves.

**Feature importance.** `src/importance.py` computes **permutation
importance** for the winning model, whichever architecture it is: each
engineered feature is shuffled in the validation set and the resulting drop
in accuracy is measured (averaged over repeats). This works identically
regardless of architecture — no gradients or model-specific internals
needed — which matters here since the winner can be the MLP or the
Transformer depending on the run.

**Benchmarking against classic SOTA baselines.**
`notebooks/model_benchmark.ipynb` is a bonus, non-graded notebook that
trains Logistic Regression, Random Forest, HistGradientBoosting, and
XGBoost on the *same* train/val split and feature engineering, for context
on how the PyTorch models compare to strong classical tabular baselines. It
is not part of the required deliverable — `train.py`'s saved model is always
one of the three PyTorch architectures above.

## Setup

```bash
git clone https://github.com/elichensaips/titanic_assigment.git
cd titanic_assigment
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Getting the data

The assignment requires fetching `train.csv` directly from Kaggle:

```bash
# 1. Get Kaggle API credentials: kaggle.com -> Account -> Settings -> API
#    Either "Create New Token" (saves kaggle.json - place at ~/.kaggle/kaggle.json)
#    or use the newer access-token flow (save the token to ~/.kaggle/access_token).
# 2. Accept the competition rules (required by Kaggle for any download):
#    https://www.kaggle.com/competitions/titanic/rules
# 3. Fetch:
python data/fetch_data.py
```

This saves `data/train.csv`. If you don't have Kaggle credentials handy,
`data/sample_train.csv` (50 rows) is included in the repo so the code can
still be inspected/run end-to-end — just pass `--data data/sample_train.csv`
to `train.py` (results will be noisier with that few rows).

## Run

**1. Train the model:**
```bash
python train.py
# optional flags: --data data/train.csv --epochs 100 --val-size 0.2 --batch-size 32 --lr 1e-3
# --archs mlp,deep_mlp,tab_transformer   (comma-separated subset to train/compare)
```
This writes `artifacts/model.pt` (the winning architecture's weights +
metadata), `artifacts/preprocessor.pkl`, `artifacts/val_split.csv`,
`artifacts/history.json`, `artifacts/model_comparison.json`, and
`artifacts/feature_importance.json`.

**2. Launch the app:**
```bash
streamlit run ds_app.py
```
- **Validation results tab** — metrics/plots on the held-out split, the
  architecture comparison table, permutation feature importance, and
  training curves, read straight from `artifacts/`.
- **Run inference tab** — point at any CSV with the Titanic schema (a file
  path or an upload). If the CSV has a `Survived` column, evaluation plots
  are shown too; otherwise you just get predictions + a download button.
  Try it against `data/sample_train.csv`.

**3. Explore the notebooks:**
```bash
jupyter notebook notebooks/eda.ipynb              # exploratory data analysis
jupyter notebook notebooks/model_benchmark.ipynb  # bonus SOTA baseline comparison
```

## Example usage

- Train: `python train.py` → for each architecture, runs 5-fold CV on the
  training split, then trains once more on the full train/val split for the
  final artifacts. Verified run on the full Kaggle `train.csv` (712 train /
  179 validation rows):

| architecture | 5-fold CV accuracy | held-out split accuracy |
|---|---|---|
| `mlp` | 83.2% ± 3.1% | 78.8% |
| `deep_mlp` | 83.3% ± 2.7% | 80.4% |
| `tab_transformer` | **85.1% ± 1.8%** (winner) | 80.4% |

  `tab_transformer` wins clearly on CV (lowest mean loss, 0.4841) — a much
  more confident margin than the single held-out split alone would suggest,
  which is exactly why CV was added for architecture selection.
- Benchmark notebook: on the same 5-fold CV, the strongest classical
  baseline was HistGradientBoosting at 83.3% — in the same band as the
  PyTorch MLPs, with the PyTorch `tab_transformer` ahead of all of them on
  this run (see the notebook's takeaways for discussion, including why its
  CV std is wider than the tree-based baselines').
- Top permutation-importance features for the winning model: `Title`, `Sex`,
  `Pclass`, `FamilySize`, `GroupSurvivalRate` — consistent with the
  "women, children, and travel-party fate" signal the EDA notebook
  identifies.
- App: `streamlit run ds_app.py` → opens at `http://localhost:8501` with
  the two tabs described above. Verified end-to-end against
  `data/sample_train.csv` (84% accuracy on that 50-row sample).

(Add screenshots of the running app here.)

## Reproducibility notes

- Random seed fixed (`--seed`, default 42) for the train/val split, model
  init, and training — and reset before training each architecture, so the
  comparison isn't confounded by different random states.
- Validation split is stratified on `Survived` to keep class balance
  consistent between train/val.
- `GroupSurvivalRate` is fit only on the training split and looked up
  (never recomputed with validation/inference labels) at transform time —
  no target leakage across the train/val boundary.
- `artifacts/` is gitignored — it's regenerated by `train.py`, not checked
  in, so the repo doesn't ship stale weights.
