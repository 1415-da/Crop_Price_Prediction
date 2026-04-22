import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
from flask import Flask, render_template, request, jsonify, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "Dataset")

METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.csv")
PRED_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
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


def metrics_plot_json(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return None
    fig = px.bar(
        metrics_df,
        x="model",
        y=["MAE", "RMSE", "R2"],
        barmode="group",
        title="Model Comparison (MAE / RMSE / R2)",
        template="plotly_white",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


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
    plot_json = metrics_plot_json(metrics_df)

    preview = pd.DataFrame(artifacts.get("sample_preview", [])).head(5)
    preview_records = preview.to_dict(orient="records") if not preview.empty else []

    return render_template(
        "index.html",
        models=["xgboost", "lightgbm", "ensemble"],
        metrics=metrics_df.to_dict(orient="records"),
        metrics_plot=plot_json,
        preview_rows=preview_records,
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

    return jsonify(
        {
            "message": "Batch prediction completed",
            "model": model_name,
            "rows": int(len(out)),
            "download_url": "/download_predictions",
            "preview": out.head(10).to_dict(orient="records"),
        }
    )


@app.route("/generate_sample", methods=["GET"])
def generate_sample():
    n = int(request.args.get("n", 10))
    model_name = request.args.get("model", "ensemble").strip().lower()
    model = models.get(model_name, models["ensemble"])

    sample_df = generate_sample_rows(n=n)
    X = prepare_input_df(sample_df)
    X_t = preprocessor.transform(X)
    preds = model.predict(X_t)

    out = sample_df.copy()
    out["predicted_price"] = np.round(preds, 2)
    out.to_csv(PRED_PATH, index=False)

    return jsonify(
        {
            "message": "Sample data generated and predicted",
            "model": model_name,
            "rows": int(len(out)),
            "download_url": "/download_predictions",
            "preview": out.head(10).to_dict(orient="records"),
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_df = get_metrics_df()
    return jsonify(
        {
            "metrics": metrics_df.to_dict(orient="records"),
            "plot": metrics_plot_json(metrics_df),
        }
    )


@app.route("/eda", methods=["GET"])
def eda():
    price_df = pd.read_csv(PRICE_PATH)
    yw_df = pd.read_csv(YIELD_PATH)

    summary = {
        "price_shape": list(price_df.shape),
        "yield_weather_shape": list(yw_df.shape),
    }
    missing = {
        "price_missing": price_df.isna().sum().sort_values(ascending=False).head(15).to_dict(),
        "yield_weather_missing": yw_df.isna().sum().sort_values(ascending=False).head(15).to_dict(),
    }

    yw_num = yw_df.select_dtypes(include=[np.number]).copy()
    corr_plot = None
    if not yw_num.empty:
        corr = yw_num.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=False,
            title="Yield/Weather Correlation Heatmap",
            aspect="auto",
            template="plotly_white",
        )
        corr_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    importances = {}
    model = models.get("xgboost")
    if hasattr(model, "feature_importances_"):
        importances = {"note": "Feature importance available on transformed feature space."}

    return jsonify(
        {
            "summary": summary,
            "missing": missing,
            "correlation_plot": corr_plot,
            "feature_importance": importances,
            "price_preview": price_df.head(10).to_dict(orient="records"),
            "yield_weather_preview": yw_df.head(10).to_dict(orient="records"),
        }
    )


@app.route("/download_predictions", methods=["GET"])
def download_predictions():
    if not os.path.exists(PRED_PATH):
        return jsonify({"error": "Prediction file not found"}), 404
    return send_file(PRED_PATH, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
