from pathlib import Path
import mlflow

BASE_DIR = Path(__file__).parent.parent.absolute()
MODEL_REGISTRY = Path(BASE_DIR, "experiments")
SAVE_DIR = Path(BASE_DIR, "saved")

MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_URI = "file:///" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(TRACKING_URI)
