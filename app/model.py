from pathlib import Path
import os, joblib, numpy as np

def resolve_model_path(default="models/clf.joblib"):
    # อนุญาต override ด้วย ENV
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()

    root = Path(__file__).resolve().parent.parent
    return (root / default).resolve()

class SkModel:
    def __init__(self, path: str | None = None):
        model_path = resolve_model_path(path or "models/clf.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = self.model.predict(X).tolist()
        probs = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1].tolist()
        return preds, probs
