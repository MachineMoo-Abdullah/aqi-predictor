from pathlib import Path
import joblib
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent.parent


def save_training_model(
        model,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test
):

    production_dir = BASE_DIR / "models" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    model.save(
        production_dir / "lstm_attention.keras"
    )

    joblib.dump(
        scaler,
        BASE_DIR / "models" / "scaler.pkl"
    )

    joblib.dump(
        X_train,
        BASE_DIR / "models" / "X_train.pkl"
    )

    joblib.dump(
        X_test,
        BASE_DIR / "models" / "X_test.pkl"
    )

    joblib.dump(
        y_train,
        BASE_DIR / "models" / "y_train.pkl"
    )

    joblib.dump(
        y_test,
        BASE_DIR / "models" / "y_test.pkl"
    )


def save_candidate(
        model,
        scaler
):

    candidate_dir = BASE_DIR / "models" / "candidate"
    history_dir = BASE_DIR / "models" / "history"

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save latest candidate model
    model.save(
        candidate_dir / "candidate.keras"
    )

    joblib.dump(
        scaler,
        candidate_dir / "scaler.pkl"
    )


    # Save version history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    model.save(
        history_dir / f"model_{timestamp}.keras"
    )

    joblib.dump(
        scaler,
        history_dir / f"scaler_{timestamp}.pkl"
    )