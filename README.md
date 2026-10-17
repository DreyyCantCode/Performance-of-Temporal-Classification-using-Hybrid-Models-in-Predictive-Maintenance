# Performance of Temporal-Classification of Hybrid Algorithm for Predicting Failure Maintained Equipment

This repository supports the thesis **"Performance of Temporal-Classification (of) Hybrid Algorithm for Predicting Failure Maintained Equipment"**.

## Goals

- Build a monitoring data pipeline that can ingest time series / event data.
- Train and evaluate models using:
  - **Bidirectional LSTM (BiLSTM)** as the first-stage sequence model
  - **XGBoost (XGB)** as the final decision model that consumes raw telemetry plus the BiLSTM sequence score
  - Raw telemetry inputs only, without additional feature-engineering transforms

## Current Architecture

The active runtime architecture is:

`Input telemetry -> BiLSTM sequence model -> BiLSTM sequence score -> XGBoost final prediction`

This means the BiLSTM runs first and its row-aligned sequence output is fed into XGBoost as part of the final classification step.

## Structure

- `data/` — place raw and processed datasets here (do not commit sensitive data).
- `src/` — core Python modules for data loading, model definitions, training, and evaluation.
- `notebooks/` — exploratory analysis and experiments.
- `models/` — saved model artifacts.

## Getting started

1. Create a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the dashboard locally:

```powershell
.\start_dashboard.ps1
```

This opens the dashboard at `http://127.0.0.1:5000/`.

4. Explore the notebook:

```bash
jupyter lab
```

5. Run training:

```powershell
.\train_model.ps1
```

Equivalent direct command:

```powershell
python -m src.train --data data/lab_maintenance_full.csv --target will_fail_24h --task classification --epochs 5
```

For time-aware evaluation (recommended for forecasting/failure prediction), run:

```powershell
.\train_model.ps1 -SplitMethod time -TimeColumn timestamp
```

If your CSV is already ordered by time, omit `-TimeColumn` and it will split chronologically by row order.

The app and training script now read directly from `data/lab_maintenance_full.csv` by default.

When training against the lab maintenance log, the pipeline derives a binary `will_fail_24h` target from the `Problem` column and uses engineered history-based features per lab and month.

## Script-First Workflow (No Notebook)

Train models from scripts:

```powershell
.\train_model.ps1
```

Run artifact-based failure prediction on asset logs:

```powershell
.\predict_assets.ps1 -DataPath data/lab_maintenance_synthetic_daily.csv -ModelDir models -OutputCsv data/predictions_debug.csv
```

Equivalent Python command:

```powershell
python -m src.predict_xgb --input-csv data/lab_maintenance_synthetic_daily.csv --model-dir models --output-csv data/predictions_debug.csv --top-n 20
```

## Notes

- This repo is a scaffold; you can adapt data loading and model definitions for your specific monitoring dataset.
- For sequence modeling, the BiLSTM expects a time-series shaped input (e.g., `(batch, timesteps, features)`).
