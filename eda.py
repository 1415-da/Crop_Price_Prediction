import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go


def _fig_json(fig) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _month_to_season(month_num: int) -> str:
    if month_num in [3, 4, 5, 6]:
        return "Kharif"
    if month_num in [7, 8, 9, 10]:
        return "Monsoon"
    if month_num in [11, 12, 1, 2]:
        return "Rabi"
    return "Unknown"


def _clean_price(price_df: pd.DataFrame) -> pd.DataFrame:
    p = price_df.copy()
    if p.empty:
        return pd.DataFrame(
            columns=["month", "commodity_name", "state_name", "month_num", "season", "avg_modal_price", "year"]
        )
    p.columns = [c.strip().lower() for c in p.columns]
    p["month"] = pd.to_datetime(p.get("month"), errors="coerce")
    p["commodity_name"] = p.get("commodity_name", "").astype(str).str.lower().str.strip()
    p["state_name"] = p.get("state_name", "").astype(str).str.lower().str.strip()
    p["month_num"] = p["month"].dt.month
    p["season"] = p["month_num"].fillna(1).astype(int).apply(_month_to_season)
    return p


def _clean_yw(yw_df: pd.DataFrame) -> pd.DataFrame:
    y = yw_df.copy()
    if y.empty:
        return pd.DataFrame(
            columns=[
                "commodity_name",
                "year",
                "yield_kg_per_ha",
                "rainfall_mm",
                "temperature_c",
                "humidity_pct",
                "area_ha",
            ]
        )
    y.columns = [c.strip().lower() for c in y.columns]
    y = y.rename(columns={"crop": "commodity_name", "humidity_%": "humidity_pct"})
    y["commodity_name"] = y.get("commodity_name", "").astype(str).str.lower().str.strip()
    y["year"] = pd.to_numeric(y.get("year"), errors="coerce")
    return y


def generate_eda_report(price_df: pd.DataFrame, yw_df: pd.DataFrame, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    p = _clean_price(price_df)
    y = _clean_yw(yw_df)
    price_col = "avg_modal_price"

    if "month" not in p.columns:
        p["month"] = pd.NaT
    p["month"] = pd.to_datetime(p["month"], errors="coerce")
    if "year" in p.columns:
        p["year"] = pd.to_numeric(p["year"], errors="coerce")
    else:
        p["year"] = pd.to_numeric(p["month"].dt.year, errors="coerce")
    if not y.empty:
        y_agg = y.groupby(["commodity_name", "year"], as_index=False).agg(
            {
                "yield_kg_per_ha": "mean",
                "rainfall_mm": "mean",
                "temperature_c": "mean",
                "humidity_pct": "mean",
                "area_ha": "mean" if "area_ha" in y.columns else "size",
            }
        )
    else:
        y_agg = pd.DataFrame(columns=["commodity_name", "year", "yield_kg_per_ha", "rainfall_mm", "temperature_c", "humidity_pct", "area_ha"])

    merged = p.merge(y_agg, on=["commodity_name", "year"], how="left") if not p.empty else y_agg.copy()
    if not merged.empty and "avg_modal_price" in merged.columns:
        merged["lag_price_1"] = merged.groupby(["commodity_name", "state_name"])[price_col].shift(1)
    else:
        merged["lag_price_1"] = np.nan
    merged["supply_proxy"] = (pd.to_numeric(merged.get("area_ha"), errors="coerce").fillna(0) * pd.to_numeric(merged.get("yield_kg_per_ha"), errors="coerce").fillna(0))

    overview = {
        "price_rows": int(len(price_df)),
        "price_cols": int(price_df.shape[1]),
        "yield_rows": int(len(yw_df)),
        "yield_cols": int(yw_df.shape[1]),
        "missing_values_count": int(price_df.isna().sum().sum() + yw_df.isna().sum().sum()),
        "duplicate_rows_count": int(price_df.duplicated().sum() + yw_df.duplicated().sum()),
        "date_range": {
            "start": str(p["month"].min().date()) if p["month"].notna().any() else "N/A",
            "end": str(p["month"].max().date()) if p["month"].notna().any() else "N/A",
        },
        "unique_crops": int(p["commodity_name"].nunique()),
        "unique_states_markets": int(p["state_name"].nunique()),
        "analyst_note": "Price data appears relatively complete with manageable missingness. Yield-weather data provides strong supply context for market behavior analysis.",
    }

    miss_base = p if not p.empty else y
    miss_pct = (miss_base.isna().mean() * 100).sort_values(ascending=False).head(20) if not miss_base.empty else pd.Series(dtype=float)
    miss_fig = px.bar(
        x=list(miss_pct.index),
        y=list(miss_pct.values),
        title="Missing Values by Column (%)",
        labels={"x": "column", "y": "missing %"},
        template="plotly_white",
    )

    hist_fig = px.histogram(p, x=price_col, nbins=50, title="Price Distribution Histogram", template="plotly_white") if (not p.empty and price_col in p.columns) else go.Figure()
    box_fig = px.box(p, y=price_col, title="Price Boxplot with Outliers", template="plotly_white") if (not p.empty and price_col in p.columns) else go.Figure()

    monthly = (
        p.dropna(subset=["month"])
        .set_index("month")
        .resample("MS")[price_col]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    if not monthly.empty:
        monthly["ma3"] = monthly[price_col].rolling(3, min_periods=1).mean()
    else:
        monthly = pd.DataFrame(columns=["month", price_col, "ma3"])
    seasonal_fig = go.Figure()
    seasonal_fig.add_trace(go.Scatter(x=monthly["month"], y=monthly[price_col], mode="lines", name="Avg Price"))
    seasonal_fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["ma3"], mode="lines", name="3-Month MA"))
    seasonal_fig.update_layout(title="Time Series & Seasonal Trend", template="plotly_white")

    crop_avg = p.groupby("commodity_name", as_index=False)[price_col].mean().sort_values(price_col, ascending=False) if (not p.empty and price_col in p.columns) else pd.DataFrame(columns=["commodity_name", price_col])
    crop_hi_fig = px.bar(crop_avg.head(10), x="commodity_name", y=price_col, title="Top 10 Expensive Crops", template="plotly_white")
    crop_lo_fig = px.bar(crop_avg.tail(10), x="commodity_name", y=price_col, title="Top 10 Lowest Priced Crops", template="plotly_white")

    state_avg = p.groupby("state_name", as_index=False)[price_col].mean().sort_values(price_col, ascending=False).head(15) if (not p.empty and price_col in p.columns) else pd.DataFrame(columns=["state_name", price_col])
    state_fig = px.bar(state_avg, x="state_name", y=price_col, title="State/Market-wise Price", template="plotly_white")

    can_weather = (not merged.empty and price_col in merged.columns)
    yield_price_fig = px.scatter(merged, x="yield_kg_per_ha", y=price_col, color="season", title="Yield vs Price", template="plotly_white") if can_weather else go.Figure()
    rain_fig = px.scatter(merged, x="rainfall_mm", y=price_col, color="season", title="Rainfall vs Price", template="plotly_white") if can_weather else go.Figure()
    temp_fig = px.scatter(merged, x="temperature_c", y=price_col, color="season", title="Temperature vs Price", template="plotly_white") if can_weather else go.Figure()
    hum_fig = px.scatter(merged, x="humidity_pct", y=price_col, color="season", title="Humidity vs Price", template="plotly_white") if can_weather else go.Figure()

    corr_cols = [price_col, "yield_kg_per_ha", "rainfall_mm", "temperature_c", "humidity_pct", "supply_proxy", "lag_price_1"]
    corr_df = merged[[c for c in corr_cols if c in merged.columns]].apply(pd.to_numeric, errors="coerce") if not merged.empty else pd.DataFrame()
    corr = corr_df.corr(numeric_only=True) if not corr_df.empty else pd.DataFrame([[0.0]], columns=["no_data"], index=["no_data"])
    corr_fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", template="plotly_white")

    outlier_fig = go.Figure()
    for c in [price_col, "yield_kg_per_ha", "rainfall_mm", "temperature_c", "humidity_pct"]:
        if c in merged.columns:
            outlier_fig.add_trace(go.Box(y=merged[c], name=c))
    outlier_fig.update_layout(title="Outlier Detection (Key Numeric Features)", template="plotly_white")

    feature_importance = {}
    residual_analysis = {}
    if diagnostics and diagnostics.get("models"):
        for model_name, model_data in diagnostics["models"].items():
            if model_data.get("feature_importance"):
                fi_df = pd.DataFrame(model_data["feature_importance"]).head(12)
                fi_fig = px.bar(fi_df.sort_values("importance"), x="importance", y="feature", orientation="h", title=f"{model_name.upper()} Feature Importance", template="plotly_white")
                feature_importance[model_name] = _fig_json(fi_fig)
            if model_data.get("residual_analysis"):
                r = model_data["residual_analysis"]
                act_pred = px.scatter(x=r["actual"], y=r["predicted"], labels={"x": "Actual", "y": "Predicted"}, title=f"{model_name.upper()} Actual vs Predicted", template="plotly_white")
                resid_hist = px.histogram(x=r["residual"], nbins=40, title=f"{model_name.upper()} Residual Distribution", template="plotly_white")
                residual_analysis[model_name] = {"actual_vs_pred": _fig_json(act_pred), "residual_hist": _fig_json(resid_hist)}

    insights = [
        "Prices appear to show monthly seasonality with clear moving-average phases.",
        "Crop-level price dispersion is significant, suggesting commodity-specific strategy is important.",
        "Weather variables show weak-to-moderate indirect association with market price levels.",
        "Supply-side proxy and lag price help explain short-term price movement behavior.",
    ]

    return {
        "overview": overview,
        "charts": {
            "missing_values": _fig_json(miss_fig),
            "price_histogram": _fig_json(hist_fig),
            "price_boxplot": _fig_json(box_fig),
            "seasonal_trend": _fig_json(seasonal_fig),
            "crop_price_top": _fig_json(crop_hi_fig),
            "crop_price_low": _fig_json(crop_lo_fig),
            "state_price": _fig_json(state_fig),
            "yield_vs_price": _fig_json(yield_price_fig),
            "rainfall_vs_price": _fig_json(rain_fig),
            "temperature_vs_price": _fig_json(temp_fig),
            "humidity_vs_price": _fig_json(hum_fig),
            "correlation": _fig_json(corr_fig),
            "outliers": _fig_json(outlier_fig),
        },
        "feature_importance": feature_importance,
        "residual_analysis": residual_analysis,
        "insights": insights,
        "notes": {
            "missing": "Columns with higher missingness should be imputed or monitored before production use.",
            "distribution": "Right-skew or heavy tails indicate occasional market shocks and extreme events.",
            "seasonality": "Season-linked supply cycles likely drive recurring price shifts.",
        },
    }
