from pathlib import Path
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)
