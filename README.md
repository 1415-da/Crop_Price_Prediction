# CropIntelli - Crop Price Prediction Dashboard

CropIntelli is a Flask + machine learning web app for crop price forecasting.  
It combines historical mandi price trends with yield and weather signals, then serves predictions through an interactive dashboard.

## What This Project Includes

- Multi-model regression inference:
  - `XGBoost`
  - `LightGBM`
  - `CatBoost`
  - `Random Forest`
  - `SVR`
  - `Ensemble (VotingRegressor)`
- Prediction input modes:
  - Manual single-entry form
  - CSV batch upload
  - Synthetic sample row generation
- Runtime analytics:
  - MAE, RMSE, R2
  - derived classification-style metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
  - confusion matrix and ROC curve plots
- Exploratory Data Analysis (EDA):
  - run on default or uploaded datasets
  - generate JSON and HTML EDA reports
- Downloadable reports and outputs from the UI

## Project Structure

- `app.py` - Flask app, prediction endpoints, metrics endpoints, EDA route, report downloads
- `train_models.py` - data preparation, feature engineering, model training/tuning, artifact export
- `eda.py` - EDA report generation
- `templates/index.html` - dashboard UI
- `models/` - serialized models + preprocessor artifacts
- `output/` - generated predictions, metrics, diagnostics, and reports
- `Dataset/` - source datasets used for training and optional EDA

## Quick Start

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train models and generate artifacts

```bash
python train_models.py
```

This command creates/updates artifacts in `models/` and diagnostics in `output/`.

### 4) Run the Flask app

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Training Pipeline (train_models.py)

The training script performs:

- loading and cleaning both source datasets
- feature engineering (month/season features, lag features, rolling trend, production/supply proxy)
- time-aware split for evaluation
- model training for all supported regressors
- hyperparameter search for CatBoost, Random Forest, and SVR
- ensemble model fitting
- export of:
  - `models/models.pkl`
  - `models/scaler.pkl`
  - `output/model_metrics.csv`
  - `output/model_diagnostics.json`

## Input Schema for Prediction

Primary expected columns (manual form / CSV):

- `commodity_name`
- `state_name`
- `month_num`
- `season`
- `rainfall_mm`
- `temperature_c`
- `humidity_pct`
- `yield_kg_per_ha`
- `area_ha`
- `production_proxy`
- `supply_proxy`
- `lag_price_1`
- `lag_price_3`
- `rolling_price_3`

The app supports common aliases (for example `crop`, `state`, `month`, `temperature`, `humidity`, `yield`, `area`) and fills missing fields with defaults where possible.

## Flask Routes

- `/` - dashboard home
- `/predict_manual` - single row prediction
- `/predict_csv` - batch prediction from uploaded CSV
- `/generate_sample` - synthetic sample generation
- `/predict_sample` - predict generated samples
- `/metrics` - latest metrics and diagnostics payload
- `/eda_analysis` - execute EDA on uploaded/default datasets
- `/download_predictions` - download `predictions.csv`
- `/download_overview` - download `data_overview.csv`
- `/download_metrics` - download metrics HTML report
- `/download_eda_report` - download EDA HTML report

## Generated Artifacts

After using training and/or dashboard workflows, you will typically see:

- `models/models.pkl`
- `models/scaler.pkl`
- `output/predictions.csv`
- `output/data_overview.csv`
- `output/model_metrics.csv`
- `output/model_diagnostics.json`
- `output/metrics_report.html`
- `output/eda_report.json`
- `output/eda_report.html`

## Troubleshooting

- **XGBoost import issue on macOS (OpenMP)**
  - run: `brew install libomp`
- **Dashboard shows stale charts or state**
  - hard refresh browser (`Cmd + Shift + R`)
  - restart `python app.py`
- **CSV upload or EDA parsing errors**
  - verify column names and data types
  - check Flask terminal logs for exact error details