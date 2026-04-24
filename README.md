# CropIntelli - Crop Price Prediction Dashboard

CropIntelli is a Flask-based ML dashboard for crop price forecasting using:

- historical crop price patterns (demand-side proxy)
- yield and weather signals (supply-side context)
- seasonal/time features (month + seasonality)

It supports manual prediction, CSV batch prediction, sample data generation, model comparison, and interactive EDA.

---

## Features

- Multiple regression models:
  - `XGBoost`
  - `LightGBM`
  - `CatBoost`
  - `Ensemble (VotingRegressor)`
- Input modes:
  - Manual form entry
  - CSV upload
  - Synthetic sample generation
- Metrics & diagnostics:
  - MAE, RMSE, R2, Accuracy, Precision, Recall, F1, AUC
  - Confusion Matrix and ROC/AUC chart
- Interactive EDA tab:
  - Upload one or both datasets
  - Generate chart-rich analyst report
- Downloadable reports:
  - Predictions CSV
  - Data overview CSV
  - Metrics HTML report
  - EDA HTML report

---

## Setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> Note (macOS): If XGBoost fails to load due to OpenMP:
>
> ```bash
> brew install libomp
> ```

### 3) Train models and generate artifacts

```bash
python train_models.py
```

### 4) Run Flask app

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

---

## Model Training Notes

`train_models.py` performs:

- data loading + cleaning for both datasets
- feature engineering (lags, rolling trend, supply proxy, seasonality)
- chronological split (time-aware holdout)
- CatBoost hyperparameter tuning via `RandomizedSearchCV`
- training of all models + ensemble
- metric and diagnostic export:
  - `output/model_metrics.csv`
  - `output/model_diagnostics.json`

---

## Web App Workflow

1. Select model (`XGBoost`, `LightGBM`, `CatBoost`, `Ensemble`)
2. Choose input mode:
   - Manual
   - CSV
   - Sample generation
3. Run prediction
4. View:
   - Prediction tab
   - Metrics tab
   - EDA tab
5. Download required reports

---

## Input Schema (for CSV Upload / Manual)

Expected feature columns:

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

If some fields are missing, defaults/imputation are applied where possible.

---

## Flask Routes

- `/` - main dashboard
- `/predict_manual` - single-row prediction
- `/predict_csv` - batch prediction from uploaded CSV
- `/generate_sample` - create synthetic sample rows
- `/predict_sample` - run prediction on generated sample
- `/metrics` - metrics + diagnostics JSON
- `/eda_analysis` - run EDA on uploaded/default datasets
- `/download_predictions` - download prediction CSV
- `/download_overview` - download current overview CSV
- `/download_metrics` - download metrics HTML report
- `/download_eda_report` - download EDA HTML report

---

## Report Downloads

- **Accuracy Report (HTML)** in Metrics tab:
  - table + confusion matrix + ROC/AUC
- **EDA Report (HTML)** in EDA tab:
  - all chart sections rendered as a visual analyst report

---

## Troubleshooting

- **Charts not updating / old UI**
  - hard refresh browser (`Cmd + Shift + R`)
  - restart Flask server

- **XGBoost import error on macOS**
  - install OpenMP (`brew install libomp`)

- **EDA upload error with single file**
  - ensure valid CSV format and required columns for that mode
  - check Flask terminal logs for malformed date/column types

---