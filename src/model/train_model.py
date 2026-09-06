import joblib

import os

os.makedirs("src/saved_models", exist_ok=True)

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

from .read_feature import read_features
from .prepare_data import add_daily_avg_aqi
from .data_processor import create_3day_sequences


df = read_features()

df = add_daily_avg_aqi(df)


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

SEQUENCE_LENGTH = 72

X, y = create_3day_sequences(
    df=df,
    feature_columns=feature_columns,
    target_column="daily_avg_AQI",
    sequence_length=SEQUENCE_LENGTH
)

print("Original X:", X.shape)
print("Original y:", y.shape)


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


def train_random_forest(
    X_train_ml,
    y_train,
    X_val_ml,
    y_val,
    existing_model_path=None
):

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

    rf, rf_val_pred = evaluate_model(
        "RANDOM FOREST",
        rf
    )

    joblib.dump(
        rf,
        "saved_models/rf_model.pkl"
    )

    return rf, rf_val_pred


import copy
import csv
from datetime import datetime

import requests
import joblib

from huggingface_hub import HfApi


PRODUCTION_MODEL_PATH = "src/saved_models/rf_production.pkl"

HISTORY_CSV_PATH = "src/saved_models/training_history.csv"



HUGGINGFACE_MODEL_URL = (
    "https://huggingface.co/abdullahadnan123/Random_Forest/"
    "resolve/main/rf_production.pkl"
)

HUGGINGFACE_REPO_ID = "abdullahadnan123/Random_Forest"

HUGGINGFACE_MODEL_FILENAME = "rf_production.pkl"

os.makedirs(
    "src/saved_models",
    exist_ok=True
)


def ensure_production_model():

    if os.path.exists(PRODUCTION_MODEL_PATH):

        print(
            f"Production model already exists: "
            f"{PRODUCTION_MODEL_PATH}"
        )

        return


    print("\n" + "=" * 60)
    print("PRODUCTION MODEL NOT FOUND LOCALLY")
    print("=" * 60)

    print(
        "Downloading production model from Hugging Face..."
    )

    print(
        f"URL: {HUGGINGFACE_MODEL_URL}"
    )


    os.makedirs(
        os.path.dirname(PRODUCTION_MODEL_PATH),
        exist_ok=True
    )


    response = requests.get(
        HUGGINGFACE_MODEL_URL,
        stream=True,
        timeout=600
    )

    response.raise_for_status()


    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    downloaded = 0


    with open(
        PRODUCTION_MODEL_PATH,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:

                f.write(chunk)

                downloaded += len(chunk)


                if total_size:

                    percent = (
                        downloaded /
                        total_size
                    ) * 100

                    print(
                        f"\rDownloaded: "
                        f"{percent:.1f}%",
                        end=""
                    )


    print("\n")

    print(
        "Production model saved to:"
    )

    print(
        PRODUCTION_MODEL_PATH
    )

def upload_production_model_to_huggingface():

    hf_token = os.getenv("HF_TOKEN")


    if not hf_token:

        print("\n" + "=" * 60)
        print("WARNING: HF_TOKEN NOT FOUND")
        print("=" * 60)

        print(
            "Production model was saved locally, "
            "but was NOT uploaded to Hugging Face."
        )

        return False


    if not os.path.exists(
        PRODUCTION_MODEL_PATH
    ):

        print(
            "\nERROR: Production model does not exist:"
        )

        print(
            PRODUCTION_MODEL_PATH
        )

        return False


    print("\n" + "=" * 60)
    print("UPLOADING UPDATED PRODUCTION MODEL")
    print("=" * 60)


    print(
        f"Model: {PRODUCTION_MODEL_PATH}"
    )

    print(
        f"Repository: {HUGGINGFACE_REPO_ID}"
    )


    api = HfApi(
        token=hf_token
    )


    api.upload_file(
        path_or_fileobj=PRODUCTION_MODEL_PATH,
        path_in_repo=HUGGINGFACE_MODEL_FILENAME,
        repo_id=HUGGINGFACE_REPO_ID,
        repo_type="model",
        commit_message="Update production Random Forest model"
    )


    print(
        "\nUpdated production model uploaded "
        "to Hugging Face successfully."
    )


    return True


_METRIC_KEYS = []

for _day in range(1, 4):

    _METRIC_KEYS += [
        f"day{_day}_r2",
        f"day{_day}_mae",
        f"day{_day}_rmse"
    ]


_METRIC_KEYS += [
    "avg_r2",
    "avg_mae",
    "avg_rmse"
]


HISTORY_FIELDNAMES = (
    ["timestamp", "metric_used", "decision", "winner"]

    + [
        f"candidate_{k}"
        for k in _METRIC_KEYS
    ]

    + [
        f"production_{k}"
        for k in _METRIC_KEYS
    ]
)


def evaluate_on_validation(
    model,
    X_val_ml,
    y_val
):

    val_pred = model.predict(
        X_val_ml
    )

    metrics = {}

    r2_list = []
    mae_list = []
    rmse_list = []


    for day in range(3):

        true = y_val[:, day]
        pred = val_pred[:, day]


        r2 = r2_score(
            true,
            pred
        )

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


        metrics[
            f"day{day+1}_r2"
        ] = r2

        metrics[
            f"day{day+1}_mae"
        ] = mae

        metrics[
            f"day{day+1}_rmse"
        ] = rmse


        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)


    metrics["avg_r2"] = float(
        np.mean(r2_list)
    )

    metrics["avg_mae"] = float(
        np.mean(mae_list)
    )

    metrics["avg_rmse"] = float(
        np.mean(rmse_list)
    )


    return metrics


def log_history(row: dict):

    os.makedirs(
        os.path.dirname(HISTORY_CSV_PATH),
        exist_ok=True
    )


    file_exists = os.path.exists(
        HISTORY_CSV_PATH
    )


    with open(
        HISTORY_CSV_PATH,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=HISTORY_FIELDNAMES
        )


        if not file_exists:

            writer.writeheader()


        writer.writerow({
            k: row.get(k, "")
            for k in HISTORY_FIELDNAMES
        })


def train_candidate(
    X_train_ml,
    y_train,
    production_model_path=PRODUCTION_MODEL_PATH
):

    ensure_production_model()


    if (
        production_model_path
        and os.path.exists(
            production_model_path
        )
    ):

        base_model = joblib.load(
            production_model_path
        )


        candidate = copy.deepcopy(
            base_model
        )


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


    candidate.fit(
        X_train_ml,
        y_train
    )


    return candidate


def compare_and_promote(
    candidate,
    X_val_ml,
    y_val,
    production_model_path=PRODUCTION_MODEL_PATH,
    metric="avg_mae",
    lower_is_better=True,
):


    candidate_metrics = evaluate_on_validation(
        candidate,
        X_val_ml,
        y_val
    )


    production_metrics = None


    if os.path.exists(
        production_model_path
    ):

        production_model = joblib.load(
            production_model_path
        )


        production_metrics = (
            evaluate_on_validation(
                production_model,
                X_val_ml,
                y_val
            )
        )


    if production_metrics is None:

        decision = (
            "promoted_candidate "
            "(no prior production model)"
        )

        winner = "candidate"


        joblib.dump(
            candidate,
            production_model_path
        )


        print(
            f"Saved new production model "
            f"-> {production_model_path}"
        )


        # Upload newly created production model
        upload_production_model_to_huggingface()


    else:

        cand_score = candidate_metrics[
            metric
        ]

        prod_score = production_metrics[
            metric
        ]


        candidate_wins = (
            cand_score < prod_score
            if lower_is_better
            else cand_score > prod_score
        )

\
        if candidate_wins:

            decision = "promoted_candidate"

            winner = "candidate"


            joblib.dump(
                candidate,
                production_model_path
            )


            print(
                f"Saved new production model "
                f"-> {production_model_path}"
            )


            # Upload updated production model
            upload_production_model_to_huggingface()


        else:

            decision = "kept_production"

            winner = "production"


            print(
                "Production model unchanged, "
                "still the best -> "
                f"{production_model_path}"
            )

    row = {

        "timestamp":
            datetime.utcnow().isoformat(),

        "metric_used":
            metric,

        "decision":
            decision,

        "winner":
            winner,

    }


    for k, v in candidate_metrics.items():

        row[
            f"candidate_{k}"
        ] = v


    if production_metrics:

        for k, v in production_metrics.items():

            row[
                f"production_{k}"
            ] = v


    log_history(
        row
    )


    prod_str = (

        f"{production_metrics[metric]:.4f}"

        if production_metrics

        else "n/a"

    )


    print(
        f"\nDecision: {decision} | "
        f"metric={metric} | "
        f"candidate="
        f"{candidate_metrics[metric]:.4f} | "
        f"production={prod_str}"
    )


    return (
        winner,
        candidate_metrics,
        production_metrics
    )


def retrain_weekly():

    print("\n" + "=" * 60)

    print(
        "WEEKLY RETRAIN — pulling latest data"
    )

    print("=" * 60)


    ensure_production_model()


    df = read_features()

    df = add_daily_avg_aqi(
        df
    )


    X, y = create_3day_sequences(

        df=df,

        feature_columns=
            feature_columns,

        target_column=
            "daily_avg_AQI",

        sequence_length=
            SEQUENCE_LENGTH

    )


    n_samples = len(X)


    train_end = int(
        n_samples * 0.70
    )


    X_train = X[
        :train_end
    ]

    y_train = y[
        :train_end
    ]


    X_val = X[
        train_end:
    ]

    y_val = y[
        train_end:
    ]


    X_train_ml = X_train.reshape(
        X_train.shape[0],
        -1
    )


    X_val_ml = X_val.reshape(
        X_val.shape[0],
        -1
    )


    candidate = train_candidate(

        X_train_ml,

        y_train,

        PRODUCTION_MODEL_PATH

    )


    winner, cand_metrics, prod_metrics = (
        compare_and_promote(

            candidate,

            X_val_ml,

            y_val,

            metric="avg_r2",

            lower_is_better=False,

        )
    )


    print(
        f"\nWeekly retrain complete. "
        f"Winner this week: {winner}"
    )


    return winner