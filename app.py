import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly
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
    metrics_df = pd.DataFrame(runtime_metrics_rows)
    diagnostics = runtime_diagnostics

    preview = pd.DataFrame(artifacts.get("sample_preview", [])).head(5)
    preview_records = preview.to_dict(orient="records") if not preview.empty else []

    return render_template(
        "index.html",
        models=["xgboost", "lightgbm", "ensemble"],
        metrics=metrics_df.to_dict(orient="records"),
        diagnostics=diagnostics,
        preview_rows=preview_records,
        runtime_source=runtime_source,
    )


@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
    df = pd.DataFrame([payload])
    X = prepare_input_df(df)
    model_name = payload.get("model", "ensemble").strip().lower()
    model = models.get(model_name, models["ensemble"])
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
    model_name = request.form.get("model", "ensemble").strip().lower()
    model = models.get(model_name, models["ensemble"])

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
    model_name = str(payload.get("model", "ensemble")).strip().lower()
    model = models.get(model_name, models["ensemble"])

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
    metrics_df = pd.DataFrame(runtime_metrics_rows)
    return jsonify(
        {
            "metrics": metrics_df.to_dict(orient="records"),
            "diagnostics": runtime_diagnostics,
            "source": runtime_source,
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


if __name__ == "__main__":
    app.run(debug=True)
