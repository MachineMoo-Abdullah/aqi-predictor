# ============================================================
# TRADITIONAL ML MODELS FOR 3-DAY AQI FORECASTING
#
# Input  : Previous 72 hours
# Output : Day 1, Day 2, Day 3 daily average AQI
#
# Split  : 85% Train / 15% Validation  (chronological, no test set)
#
# Models:
#   1. Random Forest
#   2. Extra Trees
#   3. Gradient Boosting
#   4. Ensemble
# ============================================================
import joblib

# add this once at the top, after the imports
import os
os.makedirs("saved_models", exist_ok=True)

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD YOUR DATA
# ============================================================

from .read_feature import read_features
from .prepare_data import add_daily_avg_aqi
from .data_processor import create_3day_sequences


df = read_features()

df = add_daily_avg_aqi(df)


# ============================================================
# 2. FEATURES
# ============================================================

feature_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "AQI",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]


# ============================================================
# 3. CREATE 72-HOUR SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 72

X, y = create_3day_sequences(
    df=df,
    feature_columns=feature_columns,
    target_column="daily_avg_AQI",
    sequence_length=SEQUENCE_LENGTH
)

print("Original X:", X.shape)
print("Original y:", y.shape)


# ============================================================
# 4. CHRONOLOGICAL SPLIT  ->  85% TRAIN / 15% VALIDATION
# ============================================================
# No test set - validation is the only held-out performance check.

n_samples = len(X)

train_end = int(n_samples * 0.70)


X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:]
y_val = y[train_end:]


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Train     :", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)


# ============================================================
# 5. CONVERT 3D SEQUENCES TO 2D
# ============================================================
#
# LSTM expects:
#
#     [samples, 72, 16]
#
# Random Forest expects:
#
#     [samples, features]
#
# Therefore:
#
#     72 × 16 = 1152 features
#
# ============================================================

X_train_ml = X_train.reshape(
    X_train.shape[0],
    -1
)

X_val_ml = X_val.reshape(
    X_val.shape[0],
    -1
)


print("\nML input shape:")
print("Train     :", X_train_ml.shape)
print("Validation:", X_val_ml.shape)


# ============================================================
# 6. FUNCTION TO EVALUATE MODEL
# ============================================================

def evaluate_model(name, model):

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)


    # Train
    model.fit(
        X_train_ml,
        y_train
    )


    # Validation prediction
    val_pred = model.predict(
        X_val_ml
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nVALIDATION RESULTS")

    for day in range(3):

        true = y_val[:, day]
        pred = val_pred[:, day]

        mae = mean_absolute_error(
            true,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                true,
                pred
            )
        )

        r2 = r2_score(
            true,
            pred
        )

        print(
            f"Day {day + 1}: "
            f"R²={r2:.4f} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f}"
        )


    return model, val_pred

def train_random_forest(X_train_ml, y_train, X_val_ml, y_val, existing_model_path=None):

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        warm_start=True
    )

    rf, rf_val_pred = evaluate_model("RANDOM FOREST", rf)
    # evaluate_model calls rf.fit(X_train_ml, y_train) internally

    joblib.dump(rf, "saved_models/rf_model.pkl")
    return rf, rf_val_pred


import copy
import csv
from datetime import datetime

PRODUCTION_MODEL_PATH = "saved_models/rf_production.pkl"
HISTORY_CSV_PATH = "saved_models/training_history.csv"

# Fixed column layout so the CSV never misaligns across weekly runs
_METRIC_KEYS = []
for _day in range(1, 4):
    _METRIC_KEYS += [f"day{_day}_r2", f"day{_day}_mae", f"day{_day}_rmse"]
_METRIC_KEYS += ["avg_r2", "avg_mae", "avg_rmse"]

HISTORY_FIELDNAMES = (
    ["timestamp", "metric_used", "decision", "winner"]
    + [f"candidate_{k}" for k in _METRIC_KEYS]
    + [f"production_{k}" for k in _METRIC_KEYS]
)


def evaluate_on_validation(model, X_val_ml, y_val):
    val_pred = model.predict(X_val_ml)
    metrics = {}
    r2_list, mae_list, rmse_list = [], [], []

    for day in range(3):
        true, pred = y_val[:, day], val_pred[:, day]
        r2 = r2_score(true, pred)
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        metrics[f"day{day+1}_r2"] = r2
        metrics[f"day{day+1}_mae"] = mae
        metrics[f"day{day+1}_rmse"] = rmse
        r2_list.append(r2); mae_list.append(mae); rmse_list.append(rmse)

    metrics["avg_r2"] = float(np.mean(r2_list))
    metrics["avg_mae"] = float(np.mean(mae_list))
    metrics["avg_rmse"] = float(np.mean(rmse_list))
    return metrics


def log_history(row: dict):
    file_exists = os.path.exists(HISTORY_CSV_PATH)
    with open(HISTORY_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDNAMES})


def train_candidate(X_train_ml, y_train, production_model_path=PRODUCTION_MODEL_PATH):
    """
    Builds a CANDIDATE model. Never mutates the production file directly —
    deep-copies it first, so if the candidate loses, production is untouched.
    """
    if production_model_path and os.path.exists(production_model_path):
        base_model = joblib.load(production_model_path)
        candidate = copy.deepcopy(base_model)
        candidate.warm_start = True
        candidate.n_estimators += 100
    else:
        candidate = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            warm_start=True
        )

    candidate.fit(X_train_ml, y_train)
    return candidate


def compare_and_promote(
    candidate,
    X_val_ml, y_val,
    production_model_path=PRODUCTION_MODEL_PATH,
    metric="avg_mae",
    lower_is_better=True,
):
    """
    Evaluates candidate vs. current production model on the SAME
    validation set. Promotes the winner to production_model_path.
    Every run is logged to HISTORY_CSV_PATH regardless of outcome.
    """
    candidate_metrics = evaluate_on_validation(candidate, X_val_ml, y_val)

    production_metrics = None
    if os.path.exists(production_model_path):
        production_model = joblib.load(production_model_path)
        production_metrics = evaluate_on_validation(production_model, X_val_ml, y_val)

    if production_metrics is None:
        decision, winner = "promoted_candidate (no prior production model)", "candidate"
        joblib.dump(candidate, production_model_path)
    else:
        cand_score, prod_score = candidate_metrics[metric], production_metrics[metric]
        candidate_wins = cand_score < prod_score if lower_is_better else cand_score > prod_score

        if production_metrics is None:
            decision, winner = "promoted_candidate (no prior production model)", "candidate"
            joblib.dump(candidate, production_model_path)
            print(f"Saved new production model -> {production_model_path}")
        else:
            cand_score, prod_score = candidate_metrics[metric], production_metrics[metric]
            candidate_wins = cand_score < prod_score if lower_is_better else cand_score > prod_score
    
            if candidate_wins:
                decision, winner = "promoted_candidate", "candidate"
                joblib.dump(candidate, production_model_path)
                print(f"Saved new production model -> {production_model_path}")
            else:
                decision, winner = "kept_production", "production"
                print(f"Production model unchanged, still the best -> {production_model_path}")
        

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "metric_used": metric,
        "decision": decision,
        "winner": winner,
    }
    for k, v in candidate_metrics.items():
        row[f"candidate_{k}"] = v
    if production_metrics:
        for k, v in production_metrics.items():
            row[f"production_{k}"] = v

    log_history(row)

    prod_str = f"{production_metrics[metric]:.4f}" if production_metrics else "n/a"
    print(f"\nDecision: {decision} | metric={metric} | "
          f"candidate={candidate_metrics[metric]:.4f} | production={prod_str}")

    return winner, candidate_metrics, production_metrics
    
def retrain_weekly():
    print("\n" + "=" * 60)
    print("WEEKLY RETRAIN — pulling latest data")
    print("=" * 60)

    df = read_features()                       # 1. read_feature
    df = add_daily_avg_aqi(df)                  # 2. add_daily_avg_aqi

    X, y = create_3day_sequences(               # 3. create_3day_sequences
        df=df,
        feature_columns=feature_columns,
        target_column="daily_avg_AQI",
        sequence_length=SEQUENCE_LENGTH
    )

    n_samples = len(X)
    train_end = int(n_samples * 0.70)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:], y[train_end:]

    X_train_ml = X_train.reshape(X_train.shape[0], -1)
    X_val_ml = X_val.reshape(X_val.shape[0], -1)

    # 4. build candidate, then compare against production before keeping it
    candidate = train_candidate(X_train_ml, y_train, PRODUCTION_MODEL_PATH)

    winner, cand_metrics, prod_metrics = compare_and_promote(
        candidate, X_val_ml, y_val,
        metric="avg_r2",
        lower_is_better=False,   # higher R² wins
    )
    
    print(f"\nWeekly retrain complete. Winner this week: {winner}")
    return winner
