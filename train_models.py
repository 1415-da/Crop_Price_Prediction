import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from scipy import sparse

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PRICE_PATH = os.path.join(DATA_DIR, "crop_price_dataset.csv")
YIELD_PATH = os.path.join(DATA_DIR, "Custom_Crops_yield_Historical_Dataset.csv")
DIAG_PATH = os.path.join(OUTPUT_DIR, "model_diagnostics.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_XGB_PARAMS = {
    "n_estimators": 260,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": 42,
}

BEST_LGBM_PARAMS = {
    "n_estimators": 420,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "random_state": 42,
}

BEST_CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5.0,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 0,
}

CATBOOST_SEARCH_SPACE = {
    "iterations": [300, 500, 700, 900],
    "depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "l2_leaf_reg": [3.0, 5.0, 7.0, 10.0],
    "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    "random_strength": [0.0, 0.5, 1.0, 2.0],
}


def month_to_season(month_num: int) -> str:
    if month_num in [3, 4, 5, 6]:
        return "Kharif"
    if month_num in [7, 8, 9, 10]:
        return "Monsoon"
    if month_num in [11, 12, 1, 2]:
        return "Rabi"
    return "Unknown"


def load_and_prepare_data():
    price_df = pd.read_csv(PRICE_PATH)
    yw_df = pd.read_csv(YIELD_PATH)

    price_df.columns = [c.strip().lower() for c in price_df.columns]
    price_df["month"] = pd.to_datetime(price_df["month"], errors="coerce")
    price_df = price_df.dropna(subset=["month", "commodity_name", "avg_modal_price"])
    price_df["commodity_name"] = price_df["commodity_name"].astype(str).str.strip().str.lower()
    price_df["state_name"] = price_df["state_name"].astype(str).str.strip().str.lower()
    price_df["year"] = price_df["month"].dt.year
    price_df["month_num"] = price_df["month"].dt.month
    price_df["season"] = price_df["month_num"].apply(month_to_season)
    price_df = price_df.sort_values(["commodity_name", "state_name", "month"])

    grp = price_df.groupby(["commodity_name", "state_name"])["avg_modal_price"]
    price_df["lag_price_1"] = grp.shift(1)
    price_df["lag_price_3"] = grp.shift(3)
    # Compute rolling trend per group using transform to keep index aligned across pandas versions.
    price_df["rolling_price_3"] = grp.transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    yw_df.columns = [c.strip().lower() for c in yw_df.columns]
    col_map = {
        "state name": "state_name_yw",
        "crop": "commodity_name",
        "humidity_%": "humidity_pct",
    }
    yw_df = yw_df.rename(columns=col_map)

    needed = ["year", "commodity_name", "area_ha", "yield_kg_per_ha", "temperature_c", "humidity_pct", "rainfall_mm"]
    for c in needed:
        if c not in yw_df.columns:
            yw_df[c] = np.nan

    yw_df["commodity_name"] = yw_df["commodity_name"].astype(str).str.strip().str.lower()
    yw_df["year"] = pd.to_numeric(yw_df["year"], errors="coerce")

    yw_agg = (
        yw_df.groupby(["commodity_name", "year"], as_index=False)
        .agg(
            {
                "area_ha": "mean",
                "yield_kg_per_ha": "mean",
                "temperature_c": "mean",
                "humidity_pct": "mean",
                "rainfall_mm": "mean",
            }
        )
    )

    merged = price_df.merge(yw_agg, on=["commodity_name", "year"], how="left")
    merged["production_proxy"] = merged["area_ha"] * merged["yield_kg_per_ha"]
    merged["supply_proxy"] = merged["production_proxy"] / (merged["production_proxy"].abs().max() + 1e-9)

    lag_cols = ["lag_price_1", "lag_price_3", "rolling_price_3"]
    for c in lag_cols:
        merged[c] = merged[c].fillna(merged.groupby("commodity_name")[c].transform("median"))
        merged[c] = merged[c].fillna(merged[c].median())

    target_col = "avg_modal_price"
    feature_cols = [
        "commodity_name",
        "state_name",
        "month_num",
        "season",
        "rainfall_mm",
        "temperature_c",
        "humidity_pct",
        "yield_kg_per_ha",
        "area_ha",
        "production_proxy",
        "supply_proxy",
        "lag_price_1",
        "lag_price_3",
        "rolling_price_3",
    ]

    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    model_df = merged[feature_cols + [target_col, "month"]].copy()
    model_df = model_df.dropna(subset=[target_col])
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col])
    return model_df, feature_cols, target_col


def train():
    df, feature_cols, target_col = load_and_prepare_data()
    # Chronological split gives more realistic metrics for forecasting style problems.
    df = df.sort_values("month").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Not enough data for train/test split after preprocessing.")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    categorical_cols = ["commodity_name", "state_name", "season"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(**BEST_XGB_PARAMS)
    except Exception:
        xgb = RandomForestRegressor(n_estimators=400, random_state=42)

    try:
        from lightgbm import LGBMRegressor

        lgbm = LGBMRegressor(**BEST_LGBM_PARAMS)
    except Exception:
        lgbm = RandomForestRegressor(n_estimators=500, random_state=42)

    try:
        from catboost import CatBoostRegressor

        catboost_model = CatBoostRegressor(**BEST_CATBOOST_PARAMS)
    except Exception:
        catboost_model = RandomForestRegressor(n_estimators=550, random_state=42)

    if catboost_model.__class__.__name__ == "CatBoostRegressor":
        tscv = TimeSeriesSplit(n_splits=4)
        X_train_cat = X_train_t.toarray() if sparse.issparse(X_train_t) else X_train_t
        cat_search = RandomizedSearchCV(
            estimator=catboost_model,
            param_distributions=CATBOOST_SEARCH_SPACE,
            n_iter=15,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        cat_search.fit(X_train_cat, y_train)
        catboost_model = cat_search.best_estimator_
        print(f"Tuned CatBoost best params: {cat_search.best_params_}")

    ensemble = VotingRegressor(
        estimators=[
            ("xgb", xgb),
            ("lgbm", lgbm),
        ]
    )

    models = {
        "xgboost": xgb,
        "lightgbm": lgbm,
        "catboost": catboost_model,
        "ensemble": ensemble,
    }

    metrics_rows = []
    trained_models = {}
    diagnostics = {"models": {}, "best_model": None}

    for name, model in models.items():
        X_train_model = X_train_t.toarray() if (name == "catboost" and sparse.issparse(X_train_t)) else X_train_t
        X_test_model = X_test_t.toarray() if (name == "catboost" and sparse.issparse(X_test_t)) else X_test_t
        model.fit(X_train_model, y_train)
        pred = model.predict(X_test_model)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        # Directional classification metrics: did price move up/down vs lag_price_1?
        # This converts regression output into a practical trend signal.
        actual_up = (y_test.values >= X_test["lag_price_1"].values).astype(int)
        pred_up = (pred >= X_test["lag_price_1"].values).astype(int)
        acc = accuracy_score(actual_up, pred_up)
        precision = precision_score(actual_up, pred_up, zero_division=0)
        recall = recall_score(actual_up, pred_up, zero_division=0)
        f1 = f1_score(actual_up, pred_up, zero_division=0)
        cm = confusion_matrix(actual_up, pred_up, labels=[0, 1]).tolist()
        score_for_roc = pred - X_test["lag_price_1"].values
        fpr, tpr, _ = roc_curve(actual_up, score_for_roc)
        auc_score = auc(fpr, tpr)

        metrics_rows.append(
            {
                "model": name,
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "R2": round(float(r2), 4),
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(precision), 4),
                "Recall": round(float(recall), 4),
                "F1": round(float(f1), 4),
                "AUC": round(float(auc_score), 4),
            }
        )
        trained_models[name] = model
        diagnostics["models"][name] = {
            "confusion_matrix": cm,
            "roc_curve": {
                "fpr": [float(x) for x in fpr],
                "tpr": [float(x) for x in tpr],
                "auc": float(auc_score),
            },
            "residual_analysis": {
                "actual": [float(x) for x in y_test.values[:500]],
                "predicted": [float(x) for x in pred[:500]],
                "residual": [float(x) for x in (y_test.values - pred)[:500]],
            },
        }

        if hasattr(model, "feature_importances_"):
            raw_importance = model.feature_importances_
            try:
                feature_names = preprocessor.get_feature_names_out().tolist()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(len(raw_importance))]
            pairs = sorted(
                zip(feature_names, raw_importance),
                key=lambda x: float(x[1]),
                reverse=True,
            )[:20]
            diagnostics["models"][name]["feature_importance"] = [
                {"feature": str(k), "importance": float(v)} for k, v in pairs
            ]

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE")
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    diagnostics["best_model"] = str(metrics_df.iloc[0]["model"]) if not metrics_df.empty else None
    with open(DIAG_PATH, "w", encoding="utf-8") as f:
        import json

        json.dump(diagnostics, f, ensure_ascii=True, indent=2)

    numeric_defaults = train_df[numeric_cols].median(numeric_only=True).to_dict()
    cat_defaults = {
        "commodity_name": (
            train_df["commodity_name"].mode().iloc[0]
            if not train_df["commodity_name"].mode().empty
            else "rice"
        ),
        "state_name": (
            train_df["state_name"].mode().iloc[0]
            if not train_df["state_name"].mode().empty
            else "india"
        ),
        "season": (
            train_df["season"].mode().iloc[0]
            if not train_df["season"].mode().empty
            else "Rabi"
        ),
    }
    sample_rows = train_df.head(10).copy()

    artifacts = {
        "models": trained_models,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_defaults": numeric_defaults,
        "cat_defaults": cat_defaults,
        "sample_preview": sample_rows.to_dict(orient="records"),
    }

    with open(os.path.join(MODEL_DIR, "models.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    print("\nTraining complete. Metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {os.path.join(MODEL_DIR, 'models.pkl')}")
    print(f"Saved: {os.path.join(MODEL_DIR, 'scaler.pkl')}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {DIAG_PATH}")


if __name__ == "__main__":
    train()
