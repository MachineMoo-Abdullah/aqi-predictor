from pathlib import Path
from datetime import datetime
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent.parent

HISTORY_DIR = Path(BASE_DIR/"models"/"history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = HISTORY_DIR / "metrics.csv"


def log_metrics(

    production_mae,
    production_rmse,
    production_r2,

    candidate_mae,
    candidate_rmse,
    candidate_r2

):

    accepted = candidate_rmse < production_rmse

    row = pd.DataFrame([{

        "datetime":
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "production_mae":
        production_mae,

        "production_rmse":
        production_rmse,

        "production_r2":
        production_r2,

        "candidate_mae":
        candidate_mae,

        "candidate_rmse":
        candidate_rmse,

        "candidate_r2":
        candidate_r2,

        "status":
        "Accepted" if accepted else "Rejected"

    }])

    if METRICS_FILE.exists():

        row.to_csv(

            METRICS_FILE,

            mode="a",

            header=False,

            index=False

        )

    else:

        row.to_csv(

            METRICS_FILE,

            index=False

        )

    print("\nMetrics logged successfully.")

    return accepted