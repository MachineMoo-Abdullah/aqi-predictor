import sys
from pathlib import Path
import logging
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.train_model import retrain_weekly


def run():

    try:
        retrain_weekly()
    except Exception:

        print("Weekly retrain pipeline FAILED")


if __name__ == "__main__":
    run()