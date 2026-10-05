import threading
from pathlib import Path
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "oneD_Conv.keras"

_lock = threading.Lock()
_model = None


def get_model():
    """Return the current model, loading it on first use."""
    global _model
    with _lock:
        if _model is None:
            _model = keras.models.load_model(MODEL_PATH)
        return _model


def reload_model():
    """Load the model from disk again and swap it in (call after retraining)."""
    global _model
    new_model = keras.models.load_model(MODEL_PATH)  # load first, so requests never see a half-ready model
    with _lock:
        _model = new_model
    return new_model