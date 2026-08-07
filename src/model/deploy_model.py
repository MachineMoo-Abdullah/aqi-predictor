from pathlib import Path
import shutil
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRODUCTION_DIR = Path(BASE_DIR/"models/production")
CANDIDATE_DIR = Path(BASE_DIR/"models/candidate")

PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

def deploy_model(accepted: bool):

    if not accepted:

        print("\nCandidate model rejected.")
        print("Production model kept.\n")
        return

    shutil.copy2(

        CANDIDATE_DIR / "candidate.keras",

        PRODUCTION_DIR / "lstm_attention(std).keras"

    )

    shutil.copy2(

        CANDIDATE_DIR / "scaler.pkl",

        PRODUCTION_DIR / "scaler.pkl"

    )
    print("Production model updated!")