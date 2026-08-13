# Titanic Survival Classifier

Data Science home assignment — end-to-end classification pipeline (EDA →
PyTorch training script → Streamlit evaluation/inference app) on the
[Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data).

## Architecture & design choices

```
.
├── data/
│   ├── fetch_data.py      # pulls train.csv from Kaggle via the official API
│   └── sample_train.csv   # small (50-row) sample committed to the repo
├── notebooks/
│   └── eda.ipynb          # exploratory data analysis
├── src/
│   ├── preprocessing.py   # TitanicPreprocessor: shared feature engineering
│   └── model.py           # TitanicNet: PyTorch MLP definition
├── train.py                # standalone training script
├── ds_app.py                # Streamlit app: validation results + inference UI
├── requirements.txt
└── artifacts/               # created by train.py (gitignored): model.pt,
                              # preprocessor.pkl, val_split.csv, history.json
```

**Why a shared `TitanicPreprocessor`.** The exact same feature engineering
and fitted scalers/encoders have to be used at training and inference time,
or the model silently sees a different feature distribution than it was
trained on. `src/preprocessing.py` is imported by both `train.py` (fit +
transform) and `ds_app.py` (transform-only, loaded from a pickle), so there
is a single source of truth.

**Feature engineering** (see `notebooks/eda.ipynb` for the analysis behind
each choice):
- `Title` extracted from `Name` (Mr/Mrs/Miss/Master/Rare) — captures
  age/gender/class signal `Name` itself can't be used for directly.
- `FamilySize = SibSp + Parch + 1` — survival is non-monotonic in raw
  `SibSp`/`Parch` but has a cleaner U-shape in total family size.
- `HasCabin` — a boolean flag rather than the raw (77%-missing) `Cabin`
  string.
- `PassengerId`, `Name`, `Ticket`, and raw `Cabin` are dropped as either
  identifiers or too sparse to use directly.
- Numeric features are median-imputed + standard-scaled; categorical
  features are most-frequent-imputed + one-hot encoded. The pipeline is
  **fit only on the training split** to avoid leakage into validation.

**Model.** A small 2-hidden-layer MLP (`TitanicNet`, 32→16 units, ReLU,
dropout 0.3) is enough capacity for ~15-20 tabular input features and ~700
training rows — a larger network would just overfit. Trained with
`BCEWithLogitsLoss` using a `pos_weight` to correct for the ~62/38 class
imbalance, Adam with weight decay, and early stopping on validation loss.

**Evaluation.** `train.py` holds out a stratified validation split
(`--val-size`, default 20%) *before* fitting anything, and saves it to
`artifacts/val_split.csv` so the Streamlit app can score the trained model
on data it never saw. The app reports accuracy/precision/recall/F1, a
confusion matrix, an ROC curve, and the training loss/accuracy curves.

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Getting the data

The assignment requires fetching `train.csv` directly from Kaggle:

```bash
# 1. Get an API token: kaggle.com -> Account -> Settings -> API -> Create New Token
#    Place the downloaded kaggle.json at ~/.kaggle/kaggle.json
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
```
This writes `artifacts/model.pt`, `artifacts/preprocessor.pkl`,
`artifacts/val_split.csv`, and `artifacts/history.json`.

**2. Launch the app:**
```bash
streamlit run ds_app.py
```
- **Validation results tab** — metrics/plots on the held-out split, plus
  training curves, read straight from `artifacts/`.
- **Run inference tab** — point at any CSV with the Titanic schema (a file
  path or an upload). If the CSV has a `Survived` column, evaluation plots
  are shown too; otherwise you just get predictions + a download button.
  Try it against `data/sample_train.csv`.

**3. Explore the EDA:**
```bash
jupyter notebook notebooks/eda.ipynb
```

## Example usage

- Train: `python train.py` → prints per-epoch train/val loss & accuracy,
  finishes around 82% validation accuracy on the full Kaggle `train.csv`.
- App: `streamlit run ds_app.py` → opens at `http://localhost:8501` with
  the two tabs described above.

(Add screenshots of the running app here.)

## Reproducibility notes

- Random seed fixed (`--seed`, default 42) for the train/val split, model
  init, and training.
- Validation split is stratified on `Survived` to keep class balance
  consistent between train/val.
- `artifacts/` is gitignored — it's regenerated by `train.py`, not checked
  in, so the repo doesn't ship stale weights.
