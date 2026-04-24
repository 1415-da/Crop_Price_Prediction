import os
import json
import pickle
from html import escape
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly
import plotly.io as pio
from flask import Flask, render_template, request, jsonify, send_file
from eda import generate_eda_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "Dataset")

METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.csv")
PRED_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
OVERVIEW_PATH = os.path.join(OUTPUT_DIR, "data_overview.csv")
DIAG_PATH = os.path.join(OUTPUT_DIR, "model_diagnostics.json")
EDA_REPORT_PATH = os.path.join(OUTPUT_DIR, "eda_report.json")
EDA_REPORT_HTML_PATH = os.path.join(OUTPUT_DIR, "eda_report.html")
METRICS_REPORT_HTML_PATH = os.path.join(OUTPUT_DIR, "metrics_report.html")
PRICE_PATH = os.path.join(DATA_DIR, "crop_price_dataset.csv")
YIELD_PATH = os.path.join(DATA_DIR, "Custom_Crops_yield_Historical_Dataset.csv")

app = Flask(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def month_to_season(month_num: int) -> str:
    if month_num in [3, 4, 5, 6]:
        return "Kharif"
    if month_num in [7, 8, 9, 10]:
        return "Monsoon"
    if month_num in [11, 12, 1, 2]:
        return "Rabi"
    return "Unknown"


def load_artifacts():
    with open(os.path.join(MODEL_DIR, "models.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    return artifacts, preprocessor


artifacts, preprocessor = load_artifacts()
models = artifacts["models"]
feature_cols = artifacts["feature_cols"]
numeric_cols = artifacts["numeric_cols"]
categorical_cols = artifacts["categorical_cols"]
numeric_defaults = artifacts["numeric_defaults"]
cat_defaults = artifacts["cat_defaults"]
runtime_metrics_rows = []
runtime_diagnostics = {"models": {}, "best_model": None}
runtime_source = "No input evaluated yet."
MODEL_DISPLAY_ORDER = ["xgboost", "lightgbm", "catboost", "random_forest", "svr", "ensemble"]
COLUMN_DESCRIPTIONS = {
    "crop_price_dataset.csv": {
        "month": {
            "description": "Month/date of market observation.",
            "role": "Input/Context",
        },
        "commodity_name": {
            "description": "Crop/commodity name sold in mandi.",
            "role": "Feature",
        },
        "avg_modal_price": {
            "description": "Target price; average modal mandi price.",
            "role": "Target",
        },
        "avg_min_price": {
            "description": "Average minimum mandi price in that period.",
            "role": "Input/Context",
        },
        "avg_max_price": {
            "description": "Average maximum mandi price in that period.",
            "role": "Input/Context",
        },
        "state_name": {
            "description": "State where mandi price is recorded.",
            "role": "Feature",
        },
        "district_name": {
            "description": "District/market region for the record.",
            "role": "Input/Context",
        },
        "calculationType": {
            "description": "Aggregation type used to compute value (for example Monthly).",
            "role": "Input/Context",
        },
        "change": {
            "description": "Price change versus previous period.",
            "role": "Input/Context",
        },
    },
    "Custom_Crops_yield_Historical_Dataset.csv": {
        "Dist Code": {"description": "District code identifier.", "role": "Input/Context"},
        "Year": {"description": "Agricultural year of observation.", "role": "Input/Context"},
        "State Code": {"description": "State code identifier.", "role": "Input/Context"},
        "State Name": {"description": "State name.", "role": "Input/Context"},
        "Dist Name": {"description": "District name.", "role": "Input/Context"},
        "Crop": {"description": "Crop name in yield/weather dataset.", "role": "Input/Context"},
        "Area_ha": {"description": "Area cultivated (hectares).", "role": "Feature"},
        "Yield_kg_per_ha": {"description": "Crop yield per hectare.", "role": "Feature"},
        "N_req_kg_per_ha": {"description": "Nitrogen requirement per hectare.", "role": "Input/Context"},
        "P_req_kg_per_ha": {"description": "Phosphorus requirement per hectare.", "role": "Input/Context"},
        "K_req_kg_per_ha": {"description": "Potassium requirement per hectare.", "role": "Input/Context"},
        "Total_N_kg": {"description": "Total nitrogen requirement for cultivated area.", "role": "Input/Context"},
        "Total_P_kg": {"description": "Total phosphorus requirement for cultivated area.", "role": "Input/Context"},
        "Total_K_kg": {"description": "Total potassium requirement for cultivated area.", "role": "Input/Context"},
        "Temperature_C": {"description": "Average temperature in Celsius.", "role": "Feature"},
        "Humidity_%": {"description": "Average relative humidity percentage.", "role": "Feature"},
        "pH": {"description": "Soil pH value.", "role": "Input/Context"},
        "Rainfall_mm": {"description": "Rainfall in millimeters.", "role": "Feature"},
        "Wind_Speed_m_s": {"description": "Wind speed in meters per second.", "role": "Input/Context"},
        "Solar_Radiation_MJ_m2_day": {"description": "Solar radiation intensity (MJ/m2/day).", "role": "Input/Context"},
    },
}
CALCULATED_TYPES = [
    {
        "column": "season",
        "calculation": "Mapped from month number: 3-6=Kharif, 7-10=Monsoon, 11-2=Rabi.",
    },
    {
        "column": "year",
        "calculation": "Extracted from month/date column (datetime year).",
    },
    {
        "column": "month_num",
        "calculation": "Extracted from month/date column (datetime month).",
    },
    {
        "column": "lag_price_1",
        "calculation": "Previous period modal price per commodity_name + state_name group.",
    },
    {
        "column": "lag_price_3",
        "calculation": "3-period lag modal price per commodity_name + state_name group.",
    },
    {
        "column": "rolling_price_3",
        "calculation": "Shifted rolling mean of last 3 modal prices within commodity_name + state_name group.",
    },
    {
        "column": "production_proxy",
        "calculation": "area_ha * yield_kg_per_ha.",
    },
    {
        "column": "supply_proxy",
        "calculation": "production_proxy normalized by max absolute production_proxy.",
    },
]


def prepare_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    aliases = {
        "crop": "commodity_name",
        "state": "state_name",
        "month": "month_num",
        "temperature": "temperature_c",
        "humidity": "humidity_pct",
        "yield": "yield_kg_per_ha",
        "area": "area_ha",
        "lag1": "lag_price_1",
        "lag3": "lag_price_3",
        "rolling3": "rolling_price_3",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "month_num" not in df.columns:
        df["month_num"] = 1
    df["month_num"] = pd.to_numeric(df["month_num"], errors="coerce").fillna(1).astype(int)

    if "season" not in df.columns:
        df["season"] = df["month_num"].apply(month_to_season)

    for c in feature_cols:
        if c not in df.columns:
            if c in numeric_cols:
                df[c] = numeric_defaults.get(c, 0.0)
            else:
                df[c] = cat_defaults.get(c, "unknown")

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(numeric_defaults.get(c, 0.0))

    for c in categorical_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": cat_defaults.get(c, "unknown")})
        df[c] = df[c].replace("", cat_defaults.get(c, "unknown"))

    return df[feature_cols]


def get_available_model_names():
    model_keys = list(models.keys())
    ordered = [m for m in MODEL_DISPLAY_ORDER if m in model_keys]
    remaining = sorted([m for m in model_keys if m not in ordered])
    return ordered + remaining


def resolve_model(model_name: str):
    aliases = {
        "randomforest": "random_forest",
        "random_forest_regressor": "random_forest",
        "rf": "random_forest",
        "support_vector_regressor": "svr",
        "svm": "svr",
    }
    key = str(model_name or "").strip().lower().replace(" ", "_")
    key = aliases.get(key, key)

    default_key = "ensemble" if "ensemble" in models else (next(iter(models.keys())) if models else None)
    chosen_key = key if key in models else default_key
    if chosen_key is None:
        raise ValueError("No trained models are available.")
    return chosen_key, models[chosen_key]


def get_metrics_df():
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame(columns=["model", "MAE", "RMSE", "R2"])


def get_diagnostics():
    if os.path.exists(DIAG_PATH):
        try:
            with open(DIAG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"models": {}, "best_model": None}
    return {"models": {}, "best_model": None}


def _plotly_div_from_json(fig_json: str) -> str:
    if not fig_json:
        return "<p>No chart data.</p>"
    try:
        fig = pio.from_json(fig_json)
        return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})
    except Exception:
        return "<p>Chart unavailable.</p>"


def _write_metrics_html_report(metrics_df: pd.DataFrame, diagnostics: dict, source: str) -> str:
    metrics_df = metrics_df.copy()
    for col in ["MAE", "RMSE", "R2"]:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
    if "RMSE" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("RMSE", na_position="last")

    rows = metrics_df.to_dict(orient="records")
    table_rows = "".join(
        f"<tr><td>{escape(str(r.get('model', '')))}</td>"
        f"<td>{escape(str(r.get('MAE', '-')))}</td>"
        f"<td>{escape(str(r.get('RMSE', '-')))}</td>"
        f"<td>{escape(str(r.get('R2', '-')))}</td></tr>"
        for r in rows
    )

    error_div = "<p>No error-comparison chart available.</p>"
    r2_div = "<p>No R2 chart available.</p>"
    chart_df = metrics_df.dropna(subset=["MAE", "RMSE", "R2"], how="any") if not metrics_df.empty else pd.DataFrame()
    if not chart_df.empty:
        models_u = [str(x).upper() for x in chart_df["model"].tolist()]
        error_fig = {
            "data": [
                {"y": models_u, "x": chart_df["MAE"].tolist(), "type": "bar", "orientation": "h", "name": "MAE", "marker": {"color": "#0d6efd"}},
                {"y": models_u, "x": chart_df["RMSE"].tolist(), "type": "bar", "orientation": "h", "name": "RMSE", "marker": {"color": "#fd7e14"}},
            ],
            "layout": {
                "title": "Error Comparison (MAE vs RMSE)",
                "barmode": "group",
                "xaxis": {"title": "Error (Lower is better)"},
                "yaxis": {"automargin": True},
                "margin": {"t": 60, "l": 90, "r": 20, "b": 45},
            },
        }
        r2_fig = {
            "data": [
                {
                    "x": models_u,
                    "y": chart_df["R2"].tolist(),
                    "type": "bar",
                    "marker": {"color": "#198754"},
                    "text": [f"{v:.3f}" if pd.notna(v) else "-" for v in chart_df["R2"].tolist()],
                    "textposition": "outside",
                }
            ],
            "layout": {
                "title": "R2 Comparison",
                "xaxis": {"tickangle": -25},
                "yaxis": {"title": "R2 (Higher is better)", "range": [0, 1]},
                "margin": {"t": 60, "l": 55, "r": 20, "b": 95},
            },
        }
        error_div = pio.to_html(error_fig, include_plotlyjs=False, full_html=False, config={"responsive": True})
        r2_div = pio.to_html(r2_fig, include_plotlyjs=False, full_html=False, config={"responsive": True})

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Metrics Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Inter,Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}th{{background:#f3f4f6}}</style>
</head><body>
<h2>Model Metrics Report</h2>
<p><strong>Source:</strong> {escape(source or 'N/A')}</p>
<table>
<thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R2</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
<h3 style="margin-top:24px;">Error Comparison</h3>{error_div}
<h3 style="margin-top:24px;">R2 Comparison</h3>{r2_div}
</body></html>"""
    with open(METRICS_REPORT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    return METRICS_REPORT_HTML_PATH


def _write_eda_html_report(payload: dict) -> str:
    charts = payload.get("charts", {})
    sections = [
        ("Missing Values", charts.get("missing_values")),
        ("Price Histogram", charts.get("price_histogram")),
        ("Price Boxplot", charts.get("price_boxplot")),
        ("Seasonal Trend", charts.get("seasonal_trend")),
        ("Top Priced Crops", charts.get("crop_price_top")),
        ("Lowest Priced Crops", charts.get("crop_price_low")),
        ("State/Market Price", charts.get("state_price")),
        ("Yield vs Price", charts.get("yield_vs_price")),
        ("Rainfall vs Price", charts.get("rainfall_vs_price")),
        ("Temperature vs Price", charts.get("temperature_vs_price")),
        ("Humidity vs Price", charts.get("humidity_vs_price")),
        ("Correlation Heatmap", charts.get("correlation")),
        ("Outlier Detection", charts.get("outliers")),
    ]
    chart_blocks = "".join(
        f"<h3 style='margin-top:24px'>{escape(title)}</h3>{_plotly_div_from_json(fig_json)}"
        for title, fig_json in sections
    )
    fi_blocks = "".join(
        f"<h4 style='margin-top:18px'>{escape(str(model_name).upper())}</h4>{_plotly_div_from_json(fig_json)}"
        for model_name, fig_json in (payload.get("feature_importance", {}) or {}).items()
    )
    residual_blocks = "".join(
        f"<h4 style='margin-top:18px'>{escape(str(model_name).upper())} - Actual vs Predicted</h4>"
        f"{_plotly_div_from_json((figs or {}).get('actual_vs_pred'))}"
        f"<h4 style='margin-top:18px'>{escape(str(model_name).upper())} - Residual Distribution</h4>"
        f"{_plotly_div_from_json((figs or {}).get('residual_hist'))}"
        for model_name, figs in (payload.get("residual_analysis", {}) or {}).items()
    )
    insights = payload.get("insights", [])
    insights_html = "".join(f"<li>{escape(str(x))}</li>" for x in insights)
    overview = payload.get("overview", {})

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EDA Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Inter,Arial,sans-serif;margin:24px}}</style>
</head><body>
<h2>EDA Analyst Report</h2>
<pre>{escape(json.dumps(overview, indent=2))}</pre>
{chart_blocks}
<h3 style="margin-top:24px;">Feature Importance by Model</h3>{fi_blocks or "<p>No feature-importance data available.</p>"}
<h3 style="margin-top:24px;">Residual Analysis by Model</h3>{residual_blocks or "<p>No residual-analysis data available.</p>"}
<h3 style="margin-top:24px;">Summary Insights</h3>
<ul>{insights_html}</ul>
</body></html>"""
    with open(EDA_REPORT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    return EDA_REPORT_HTML_PATH


def _pick_target_column(df: pd.DataFrame):
    candidates = ["avg_modal_price", "actual_price", "target", "price"]
    lower_to_actual = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_to_actual:
            return lower_to_actual[c]
    return None


def compute_runtime_metrics(input_df: pd.DataFrame, source_label: str):
    global runtime_metrics_rows, runtime_diagnostics, runtime_source

    if input_df is None or input_df.empty:
        runtime_metrics_rows = []
        runtime_diagnostics = {"models": {}, "best_model": None}
        runtime_source = source_label
        return

    X = prepare_input_df(input_df)
    target_col = _pick_target_column(input_df)
    y_true = None
    used_proxy_target = False
    if target_col:
        y_true = pd.to_numeric(input_df[target_col], errors="coerce")
        valid_mask = y_true.notna()
        if valid_mask.sum() > 0:
            X_eval = X.loc[valid_mask]
            y_true = y_true.loc[valid_mask].values
        else:
            X_eval = X
            y_true = None
    else:
        X_eval = X

    if y_true is None:
        # Fallback proxy target to avoid blank metrics for manual/generated input.
        # This uses lag_price_1 as a demand proxy target when true market price is absent.
        if "lag_price_1" in X_eval.columns:
            proxy = pd.to_numeric(X_eval["lag_price_1"], errors="coerce")
            proxy = proxy.fillna(proxy.median() if not proxy.dropna().empty else 0.0)
            y_true = proxy.values
            used_proxy_target = True
        else:
            y_true = np.zeros(len(X_eval), dtype=float)
            used_proxy_target = True

    rows = []
    diags = {"models": {}, "best_model": None}
    for name, model in models.items():
        X_t = preprocessor.transform(X_eval)
        pred = model.predict(X_t)
        row = {
            "model": name,
            "MAE": None,
            "RMSE": None,
            "R2": None,
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1": None,
            "AUC": None,
            "AvgPredPrice": round(float(np.mean(pred)), 2),
        }

        if y_true is not None and len(y_true) == len(pred):
            mae = mean_absolute_error(y_true, pred)
            rmse = np.sqrt(mean_squared_error(y_true, pred))
            r2 = r2_score(y_true, pred) if len(y_true) > 1 else np.nan
            if "lag_price_1" in X_eval.columns:
                baseline = pd.to_numeric(X_eval["lag_price_1"], errors="coerce").fillna(np.nanmedian(y_true)).values
            else:
                baseline = np.full_like(y_true, float(np.nanmedian(y_true)), dtype=float)
            actual_up = (y_true >= baseline).astype(int)
            pred_up = (pred >= baseline).astype(int)
            acc = accuracy_score(actual_up, pred_up)
            precision = precision_score(actual_up, pred_up, zero_division=0)
            recall = recall_score(actual_up, pred_up, zero_division=0)
            f1 = f1_score(actual_up, pred_up, zero_division=0)
            cm = confusion_matrix(actual_up, pred_up, labels=[0, 1]).tolist()
            if len(np.unique(actual_up)) > 1:
                score_for_roc = pred - baseline
                fpr, tpr, _ = roc_curve(actual_up, score_for_roc)
                auc_score = auc(fpr, tpr)
            else:
                # Avoid warnings/crashes when only one class exists in current input.
                fpr, tpr, auc_score = [0.0, 1.0], [0.0, 1.0], 0.5

            row.update(
                {
                    "MAE": round(float(mae), 4),
                    "RMSE": round(float(rmse), 4),
                    "R2": round(float(r2), 4) if not np.isnan(r2) else None,
                    "Accuracy": round(float(acc), 4),
                    "Precision": round(float(precision), 4),
                    "Recall": round(float(recall), 4),
                    "F1": round(float(f1), 4),
                    "AUC": round(float(auc_score), 4),
                }
            )
            diags["models"][name] = {
                "confusion_matrix": cm,
                "roc_curve": {
                    "fpr": [float(x) for x in fpr],
                    "tpr": [float(x) for x in tpr],
                    "auc": float(auc_score),
                },
            }
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    if "RMSE" in metrics_df.columns and metrics_df["RMSE"].notna().any():
        metrics_df = metrics_df.sort_values("RMSE", na_position="last")
        diags["best_model"] = str(metrics_df.iloc[0]["model"])
    else:
        diags["best_model"] = str(metrics_df.iloc[0]["model"]) if not metrics_df.empty else None

    runtime_metrics_rows = metrics_df.to_dict(orient="records")
    runtime_diagnostics = diags
    runtime_source = f"{source_label} (proxy target)" if used_proxy_target else source_label


def generate_sample_rows(n=10):
    np.random.seed(42)
    rows = []
    crop = cat_defaults.get("commodity_name", "rice")
    state = cat_defaults.get("state_name", "india")

    for _ in range(n):
        month_num = int(np.random.randint(1, 13))
        row = {
            "commodity_name": crop,
            "state_name": state,
            "month_num": month_num,
            "season": month_to_season(month_num),
            "rainfall_mm": float(np.random.uniform(200, 1600)),
            "temperature_c": float(np.random.uniform(15, 38)),
            "humidity_pct": float(np.random.uniform(35, 90)),
            "yield_kg_per_ha": float(np.random.uniform(300, 4500)),
            "area_ha": float(np.random.uniform(1000, 800000)),
            "production_proxy": float(np.random.uniform(5e5, 3e8)),
            "supply_proxy": float(np.random.uniform(0.1, 1.0)),
            "lag_price_1": float(np.random.uniform(1000, 5000)),
            "lag_price_3": float(np.random.uniform(1000, 5000)),
            "rolling_price_3": float(np.random.uniform(1000, 5000)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


@app.route("/")
def home():
    metrics_df = get_metrics_df()
    diagnostics = get_diagnostics()

    preview = pd.DataFrame(artifacts.get("sample_preview", [])).head(5)
    preview_records = preview.to_dict(orient="records") if not preview.empty else []

    return render_template(
        "index.html",
        models=get_available_model_names(),
        metrics=metrics_df.to_dict(orient="records"),
        diagnostics=diagnostics,
        preview_rows=preview_records,
        column_descriptions=COLUMN_DESCRIPTIONS,
        calculated_types=CALCULATED_TYPES,
        runtime_source="Training dataset (model_metrics.csv)",
    )


@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
    df = pd.DataFrame([payload])
    X = prepare_input_df(df)
    model_name, model = resolve_model(payload.get("model", "ensemble"))
    X_t = preprocessor.transform(X)
    pred = model.predict(X_t)
    compute_runtime_metrics(df, "Manual input")

    return jsonify(
        {
            "model": model_name,
            "prediction": round(float(pred[0]), 2),
            "input": X.to_dict(orient="records")[0],
        }
    )


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    model_name, model = resolve_model(request.form.get("model", "ensemble"))

    df = pd.read_csv(f)
    X = prepare_input_df(df)
    X_t = preprocessor.transform(X)
    preds = model.predict(X_t)

    out = df.copy()
    out["predicted_price"] = np.round(preds, 2)
    out.to_csv(PRED_PATH, index=False)
    df.to_csv(OVERVIEW_PATH, index=False)
    compute_runtime_metrics(df, "CSV upload")

    return jsonify(
        {
            "message": "Batch prediction completed",
            "model": model_name,
            "rows": int(len(out)),
            "download_url": "/download_predictions",
            "input_preview": df.head(10).to_dict(orient="records"),
            "predictions": out.head(25).to_dict(orient="records"),
            "preview": out.head(10).to_dict(orient="records"),
        }
    )


@app.route("/generate_sample", methods=["GET"])
def generate_sample():
    n = int(request.args.get("n", 10))
    sample_df = generate_sample_rows(n=n)
    sample_df.to_csv(OVERVIEW_PATH, index=False)

    return jsonify(
        {
            "message": "Sample data generated",
            "rows": int(len(sample_df)),
            "sample_data": sample_df.to_dict(orient="records"),
            "preview": sample_df.head(10).to_dict(orient="records"),
        }
    )


@app.route("/predict_sample", methods=["POST"])
def predict_sample():
    payload = request.get_json(force=True) if request.is_json else {}
    sample_data = payload.get("sample_data", [])
    model_name, model = resolve_model(payload.get("model", "ensemble"))

    if not sample_data:
        return jsonify({"error": "No sample data provided"}), 400

    sample_df = pd.DataFrame(sample_data)
    X = prepare_input_df(sample_df)
    X_t = preprocessor.transform(X)
    preds = model.predict(X_t)

    out = sample_df.copy()
    out["predicted_price"] = np.round(preds, 2)
    out.to_csv(PRED_PATH, index=False)
    compute_runtime_metrics(sample_df, "Generated sample")

    return jsonify(
        {
            "message": "Sample prediction completed",
            "model": model_name,
            "rows": int(len(out)),
            "download_url": "/download_predictions",
            "predictions": out[
                ["commodity_name", "state_name", "month_num", "season", "predicted_price"]
            ].head(25).to_dict(orient="records"),
            "preview": out.head(10).to_dict(orient="records"),
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_df = get_metrics_df()
    return jsonify(
        {
            "metrics": metrics_df.to_dict(orient="records"),
            "diagnostics": get_diagnostics(),
            "source": "Training dataset (model_metrics.csv)",
        }
    )


@app.route("/eda_analysis", methods=["POST", "GET"])
def eda_analysis():
    diagnostics = get_diagnostics()
    if request.method == "POST":
        price_file = request.files.get("price_file")
        yield_file = request.files.get("yield_file")
        if price_file is not None and yield_file is not None:
            mode = "combined"
            price_df = pd.read_csv(price_file)
            yield_df = pd.read_csv(yield_file)
        elif price_file is not None:
            mode = "price_only"
            price_df = pd.read_csv(price_file)
            yield_df = pd.DataFrame()
        elif yield_file is not None:
            mode = "yield_weather_only"
            price_df = pd.DataFrame()
            yield_df = pd.read_csv(yield_file)
        else:
            mode = "default_combined"
            price_df = pd.read_csv(PRICE_PATH)
            yield_df = pd.read_csv(YIELD_PATH)
    else:
        mode = "default_combined"
        price_df = pd.read_csv(PRICE_PATH)
        yield_df = pd.read_csv(YIELD_PATH)

    payload = generate_eda_report(price_df, yield_df, diagnostics)
    payload["mode"] = mode
    with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    _write_eda_html_report(payload)
    return jsonify(payload)


@app.route("/download_predictions", methods=["GET"])
def download_predictions():
    if not os.path.exists(PRED_PATH):
        return jsonify({"error": "Prediction file not found"}), 404
    return send_file(PRED_PATH, as_attachment=True)


@app.route("/download_overview", methods=["GET"])
def download_overview():
    if not os.path.exists(OVERVIEW_PATH):
        return jsonify({"error": "Data overview file not found"}), 404
    return send_file(OVERVIEW_PATH, as_attachment=True)


@app.route("/download_metrics", methods=["GET"])
def download_metrics():
    metrics_df = get_metrics_df()
    if metrics_df.empty:
        return jsonify({"error": "Metrics report not found"}), 404
    report_path = _write_metrics_html_report(metrics_df, get_diagnostics(), "Training dataset (model_metrics.csv)")
    return send_file(report_path, as_attachment=True)


@app.route("/download_eda_report", methods=["GET"])
def download_eda_report():
    if not os.path.exists(EDA_REPORT_PATH):
        diagnostics = get_diagnostics()
        default_price = pd.read_csv(PRICE_PATH)
        default_yield = pd.read_csv(YIELD_PATH)
        payload = generate_eda_report(default_price, default_yield, diagnostics)
        payload["mode"] = "default_combined"
        with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
    with open(EDA_REPORT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    _write_eda_html_report(payload)
    return send_file(EDA_REPORT_HTML_PATH, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
