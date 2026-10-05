"""
fed_weights.py
Located at: cardiogram/src/utilities/fed_weights.py

Reads and writes the last-layer weights of oneD_Conv.keras.
"""

import numpy as np
from tensorflow import keras

from utilities.client_model import MODEL_PATH, reload_model


def get_para() -> list[np.ndarray]:
    """Return the last layer's weights, [kernel, bias], from the model on disk."""
    model = keras.models.load_model(MODEL_PATH)
    return model.layers[-1].get_weights()


def update_para(weights: list[np.ndarray]) -> None:
    """
    Overwrite the last layer of oneD_Conv.keras with the aggregated weights,
    save the model, and hot-swap it into the running app.
    """
    # Work on a fresh copy so the live model is never half-updated
    model = keras.models.load_model(MODEL_PATH)

    head = model.layers[-1]
    if not isinstance(head, keras.layers.Dense):
        raise RuntimeError(f"Expected last layer to be Dense, got {type(head).__name__}")

    current = head.get_weights()
    if len(weights) != len(current) or any(
        np.shape(w) != c.shape for w, c in zip(weights, current)
    ):
        raise ValueError(
            "Aggregated weights don't match the last layer: "
            f"got {[np.shape(w) for w in weights]}, "
            f"expected {[c.shape for c in current]}"
        )

    head.set_weights([np.asarray(w, dtype=np.float32) for w in weights])
    model.save(MODEL_PATH)
    reload_model()  # swap the new model into client_model
    print("Global weights applied and model reloaded.")