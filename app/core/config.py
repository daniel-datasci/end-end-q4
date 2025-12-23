import os

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:/home/site/wwwroot/mlruns"
)

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN is not set")
