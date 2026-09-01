
import tensorflow as tf
import joblib
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from metrics_logger import log_metrics
from deploy_model import deploy_model
BASE_DIR = Path(__file__).resolve().parent.parent.parent

PRODUCTION_MODEL = Path(
    BASE_DIR/"models/production/lstm_attention.keras"
)

CANDIDATE_MODEL = Path(
    BASE_DIR/"models/candidate/candidate.keras"
)

X_TEST = Path(
    BASE_DIR/"models/X_test.pkl"
)

Y_TEST = Path(
    BASE_DIR/"models/y_test.pkl"
)


def evaluate(model_path, X_test, y_test):

    model = tf.keras.models.load_model(model_path)

    pred = model.predict(
        X_test,
        verbose=0
    ).flatten()

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    return mae, rmse, r2


def evaluate_model():

    X_test = joblib.load(X_TEST)
    y_test = joblib.load(Y_TEST)

    print("\nEvaluating Production Model...\n")

    production_mae, production_rmse, production_r2 = evaluate(
        PRODUCTION_MODEL,
        X_test,
        y_test
    )

    print("Production Model")
    print("---------------------------")
    print(f"MAE  : {production_mae:.4f}")
    print(f"RMSE : {production_rmse:.4f}")
    print(f"R²   : {production_r2:.4f}")

    # ----------------------------------------------------
    # Initial Training
    # ----------------------------------------------------
    if not CANDIDATE_MODEL.exists():

        print("\nNo candidate model found.")
        print("Initial training completed.")
        print("Skipping model comparison.\n")

        return {
            "mae": production_mae,
            "rmse": production_rmse,
            "r2": production_r2
        }

    # ----------------------------------------------------
    # Incremental Training
    # ----------------------------------------------------
    print("\nEvaluating Candidate Model...\n")

    candidate_mae, candidate_rmse, candidate_r2 = evaluate(
        CANDIDATE_MODEL,
        X_test,
        y_test
    )

    print("Candidate Model")
    print("---------------------------")
    print(f"MAE  : {candidate_mae:.4f}")
    print(f"RMSE : {candidate_rmse:.4f}")
    print(f"R²   : {candidate_r2:.4f}")

    accepted = log_metrics(

        production_mae=production_mae,
        production_rmse=production_rmse,
        production_r2=production_r2,

        candidate_mae=candidate_mae,
        candidate_rmse=candidate_rmse,
        candidate_r2=candidate_r2
    )

    deploy_model(accepted)

    if accepted:
        print("\n✅ Candidate model deployed.")
    else:
        print("\n❌ Production model retained.")

    return accepted


