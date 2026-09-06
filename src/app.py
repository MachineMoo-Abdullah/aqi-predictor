"""
streamlit_app.py

Shows:
  1. Next 3-day average AQI forecast, using the current PRODUCTION model.
  2. At the bottom of the page: the latest logged evaluation metrics
     (candidate vs. production) from training_history.csv.

Run with:
    streamlit run streamlit_app.py
"""

import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Adjust this import to match your actual project layout
# (e.g. "from src.model.train_model import ..." if that's where it lives)
from model.train_model import (
    read_features,
    add_daily_avg_aqi,
    feature_columns,
    SEQUENCE_LENGTH,
    PRODUCTION_MODEL_PATH,
    HISTORY_CSV_PATH,
)


st.set_page_config(page_title="AQI 3-Day Forecast", layout="centered")


# ============================================================
# AQI CATEGORY HELPER (for coloring the forecast cards)
# ============================================================

def aqi_category(aqi_value: float):
    if aqi_value <= 50:
        return "Good", "#00E400"
    elif aqi_value <= 100:
        return "Moderate", "#FFFF00"
    elif aqi_value <= 150:
        return "Unhealthy (Sensitive Groups)", "#FF7E00"
    elif aqi_value <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"


# ============================================================
# HEALTH RECOMMENDATIONS (EPA-style guidance per category)
# ============================================================

HEALTH_RECOMMENDATIONS = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities as usual.",
    "Moderate": "Acceptable air quality. Unusually sensitive people should "
                "consider limiting prolonged outdoor exertion.",
    "Unhealthy (Sensitive Groups)": "People with asthma, heart or lung disease, "
                "older adults, and children should limit prolonged outdoor exertion.",
    "Unhealthy": "Everyone may begin to experience health effects; sensitive "
                 "groups may experience more serious effects. Limit prolonged "
                 "outdoor exertion.",
    "Very Unhealthy": "Health alert: everyone may experience more serious "
                      "health effects. Avoid prolonged outdoor exertion.",
    "Hazardous": "Health emergency: the entire population is more likely to "
                 "be affected. Avoid all outdoor physical activity.",
}


# ============================================================
# FEATURE NAMES FOR THE FLATTENED (72 x 16 -> 1152) INPUT
# ============================================================
# Reconstructs which raw feature + hour-offset each of the 1152
# flattened columns corresponds to, so SHAP output is readable
# instead of showing "feature 743".

def build_flat_feature_names():
    return [
        f"{feat}_t-{SEQUENCE_LENGTH - 1 - t}h"
        for t in range(SEQUENCE_LENGTH)
        for feat in feature_columns
    ]


FLAT_FEATURE_NAMES = None  # populated on first load, see loader below


# ============================================================
# LOAD PRODUCTION MODEL + SHAP EXPLAINER TOGETHER
# ============================================================
# Cached as ONE unit deliberately: the production model is grown
# weekly via warm_start, so an explainer cached separately from the
# model could silently go stale (explaining last week's tree count
# against this week's model). Bundling them means busting the cache
# for one busts it for both.

@st.cache_resource
def load_production_model_and_explainer():
    import os
    if not os.path.exists(PRODUCTION_MODEL_PATH):
        return None, None

    model = joblib.load(PRODUCTION_MODEL_PATH)

    import shap
    explainer = shap.TreeExplainer(model)

    return model, explainer


def explain_prediction(explainer, X_input_row, day_idx: int, feature_names, top_n: int = 10):
    """
    Returns a SHAP Explanation object for one prediction, one forecast day,
    ready to hand to shap.plots.waterfall().

    explainer   : shap.TreeExplainer built on the production model
    X_input_row : 1D array of length 1152 (a single flattened 72x16 sample)
    day_idx     : 0, 1, or 2 -> which of the 3 forecast days to explain
    feature_names : list of length 1152, e.g. from build_flat_feature_names()
    top_n       : how many top-contributing features to keep for display
    """
    import shap

    X_input_row = np.asarray(X_input_row).reshape(1, -1)  # (1, 1152)

    shap_values = explainer.shap_values(X_input_row)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        # Older SHAP versions: list of (n_samples, n_features) arrays,
        # one per output day.
        values = shap_values[day_idx][0]
        base_value = expected_value[day_idx]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Newer SHAP versions (e.g. 0.44+): a single array of shape
        # (n_samples, n_features, n_outputs) for multi-output models.
        values = shap_values[0, :, day_idx]
        base_value = (
            expected_value[day_idx]
            if hasattr(expected_value, "__len__")
            else expected_value
        )

    else:
        # Single-output model (e.g. one chained model per day).
        values = shap_values[0]
        base_value = (
            expected_value[0] if hasattr(expected_value, "__len__") else expected_value
        )

    values = np.asarray(values).reshape(-1)  # guard: always flat, length 1152

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_input_row[0],
        feature_names=feature_names,
    )

    # Keep only the top_n most impactful features for a readable plot —
    # SHAP's own .abs summing across 1152 features otherwise renders unusably.
    order = np.argsort(np.abs(values))[::-1][:top_n]
    return explanation[order]


# ============================================================
# BUILD THE LATEST 72-HOUR INPUT SEQUENCE FOR INFERENCE
# ============================================================

@st.cache_data(ttl=3600)  # refresh at most once an hour
def get_latest_sequence():
    df = read_features()
    df = add_daily_avg_aqi(df)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} rows of feature history, "
            f"got {len(df)}."
        )

    latest_window = df[feature_columns].iloc[-SEQUENCE_LENGTH:].values  # (72, 16)
    latest_window_ml = latest_window.reshape(1, -1)                     # (1, 72*16)

    last_timestamp = df.index[-1] if df.index.name else None

    # Recent actual daily-average AQI (last 14 days) for the trend chart.
    # Downsample hourly rows to one value per 24h using the trailing window.
    recent_hours = min(len(df), 14 * 24)
    recent_daily = (
        df["daily_avg_AQI"].iloc[-recent_hours:]
        .groupby(np.arange(recent_hours) // 24)
        .mean()
        .reset_index(drop=True)
    )

    return latest_window_ml, last_timestamp, recent_daily


# ============================================================
# PAGE — FORECAST SECTION
# ============================================================

st.title("🌫️ Next 3-Day AQI Forecast")

import os
from datetime import datetime

model, explainer = load_production_model_and_explainer()

# Load latest logged MAE per day (if available) to show as an accuracy
# disclosure next to each forecast — Day 3 is typically the least reliable.
day_mae = {1: None, 2: None, 3: None}
if os.path.exists(HISTORY_CSV_PATH):
    _hist = pd.read_csv(HISTORY_CSV_PATH)
    if not _hist.empty:
        _latest_row = _hist.iloc[-1]
        _prefix = "candidate" if _latest_row["winner"] == "candidate" else "production"
        for _d in range(1, 4):
            col_name = f"{_prefix}_day{_d}_mae"
            if col_name in _latest_row:
                day_mae[_d] = float(_latest_row[col_name])

if model is None:
    st.error(
        f"No production model found at `{PRODUCTION_MODEL_PATH}`. "
        "Run the weekly training pipeline at least once first."
    )
else:
    try:
        X_latest, last_timestamp, recent_daily = get_latest_sequence()
        pred = model.predict(X_latest)[0]  # shape (3,)

        caption_parts = []
        if last_timestamp is not None:
            caption_parts.append(f"Forecast based on data through: {last_timestamp}")
        caption_parts.append(f"App data last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.caption("  |  ".join(caption_parts))

        cols = st.columns(3)
        for i, col in enumerate(cols):
            day_num = i + 1
            aqi_val = float(pred[i])
            category, color = aqi_category(aqi_val)
            mae = day_mae.get(day_num)
            uncertainty_note = f"±{mae:.0f} (typical error)" if mae is not None else ""

            with col:
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 12px;
                        padding: 20px 10px;
                        text-align: center;
                        background-color: {color}22;
                        border: 2px solid {color};
                    ">
                        <div style="font-size: 14px; color: gray;">Day {day_num}</div>
                        <div style="font-size: 32px; font-weight: bold;">{aqi_val:.0f}</div>
                        <div style="font-size: 13px;">{category}</div>
                        <div style="font-size: 11px; color: gray;">{uncertainty_note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # --------------------------------------------------------
        # SHAP explanation — "why" behind each day's forecast.
        # Only shown if the explainer loaded successfully.
        # --------------------------------------------------------
        if explainer is not None:
            with st.expander("Why these numbers? (SHAP explanation)"):
                if FLAT_FEATURE_NAMES is None:
                    FLAT_FEATURE_NAMES = build_flat_feature_names()

                shap_day = st.radio(
                    "Explain which day?",
                    options=[1, 2, 3],
                    horizontal=True,
                    key="shap_day_selector",
                )

                explanation = explain_prediction(
                    explainer,
                    X_latest[0],
                    day_idx=shap_day - 1,
                    feature_names=FLAT_FEATURE_NAMES,
                    top_n=10,
                )

                import matplotlib.pyplot as plt
                import shap

                fig = plt.figure()
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.caption(
                    "Positive bars push the prediction higher than the model's "
                    "average output; negative bars pull it lower. Shows the "
                    "top 10 contributing features out of 1152 inputs."
                )

        # --------------------------------------------------------
        # Health recommendation — based on the worst (highest AQI)
        # of the 3 forecast days, since that's the binding constraint
        # for someone planning across the next 3 days.
        # --------------------------------------------------------
        worst_category, worst_color = aqi_category(float(pred.max()))
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {worst_color};
                padding: 10px 15px;
                margin-top: 10px;
                background-color: {worst_color}11;
            ">
                <b>Health guidance:</b> {HEALTH_RECOMMENDATIONS[worst_category]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # Historical trend + forecast, combined on one chart.
        #
        # Built with Altair directly (not st.line_chart) because
        # st.line_chart treats string x-labels ("-14d", "-1d", "Day 1")
        # as a nominal field and SORTS THEM ALPHABETICALLY by default —
        # that scrambles the real time order and breaks the visual
        # connection between the actual and forecast lines. Here we
        # keep a numeric "step" column and sort explicitly by it.
        # --------------------------------------------------------
        st.subheader("Recent trend + forecast")

        n_actual = len(recent_daily)
        index_labels = [f"-{n_actual - i}d" for i in range(n_actual)] + ["Day 1", "Day 2", "Day 3"]
        steps = list(range(n_actual + 3))

        actual_series = list(recent_daily.values) + [None, None, None]
        # Forecast series repeats the last actual value at the boundary
        # step so the two lines share a point and connect with no gap.
        forecast_series = [None] * (n_actual - 1) + [recent_daily.values[-1]] + list(pred)

        long_df = pd.DataFrame({
            "step": steps * 2,
            "label": index_labels * 2,
            "AQI": actual_series + forecast_series,
            "Type": ["Actual"] * len(actual_series) + ["Forecast"] * len(forecast_series),
        }).dropna(subset=["AQI"])

        chart = (
            alt.Chart(long_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("label:N", sort=alt.SortField(field="step", order="ascending"), title=""),
                y=alt.Y("AQI:Q", title="AQI"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(domain=["Actual", "Forecast"], range=["#83c9ff", "#ff8700"]),
                    legend=alt.Legend(title=""),
                ),
                strokeDash=alt.StrokeDash(
                    "Type:N",
                    scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [5, 3]]),
                    legend=None,
                ),
                tooltip=["label", "Type", alt.Tooltip("AQI:Q", format=".0f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate forecast: {e}")


# ============================================================
# PAGE — LATEST MODEL EVALUATION METRICS (bottom of page)
# ============================================================

st.divider()
st.header("📊 Latest Model Evaluation")

import os
if not os.path.exists(HISTORY_CSV_PATH):
    st.info("No training history logged yet.")
else:
    history_df = pd.read_csv(HISTORY_CSV_PATH)

    if history_df.empty:
        st.info("Training history file exists but has no rows yet.")
    else:
        latest = history_df.iloc[-1]

        st.caption(f"Run timestamp: {latest['timestamp']}  |  "
                   f"Decision: **{latest['decision']}**  |  "
                   f"Metric used: `{latest['metric_used']}`")

        winner_prefix = "candidate" if latest["winner"] == "candidate" else "production"

        metric_rows = []
        for day in range(1, 4):
            metric_rows.append({
                "Day": f"Day {day}",
                "R²": latest[f"{winner_prefix}_day{day}_r2"],
                "MAE": latest[f"{winner_prefix}_day{day}_mae"],
                "RMSE": latest[f"{winner_prefix}_day{day}_rmse"],
            })
        metric_rows.append({
            "Day": "Average",
            "R²": latest[f"{winner_prefix}_avg_r2"],
            "MAE": latest[f"{winner_prefix}_avg_mae"],
            "RMSE": latest[f"{winner_prefix}_avg_rmse"],
        })

        st.subheader(f"Serving model metrics ({winner_prefix})")
        st.dataframe(
            pd.DataFrame(metric_rows).set_index("Day").style.format("{:.4f}"),
            use_container_width=True,
        )
