import joblib
from pathlib import Path

BASE_PATH = Path("artifacts/models")

def load_model(name):
    return joblib.load(BASE_PATH / f"{name}.pkl")
