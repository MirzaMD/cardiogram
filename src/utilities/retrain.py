"""
retrain.py
Located at: cardiogram/src/utilities/retrain.py

Retrains ONLY the last layer (the softmax Dense) of oneD_Conv.keras using the
labeled ECG windows stored in user_details.db, then saves the model back.

Run from the `src` folder:
    python -m utilities.retrain
"""
from __future__ import annotations
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
from tensorflow import keras
import os

EDGE_DEVICE_ID = int(os.environ.get("EDGE_DEVICE_ID", 1))  # default to 1 if not set
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "oneD_Conv.keras"
BACKUP_PATH = PROJECT_ROOT / "oneD_Conv.backup.keras"
DB_PATH = PROJECT_ROOT / "src" / "user_details.db"

WINDOW_SIZE = 180


def load_training_data(
    db_path: str | Path = DB_PATH, n_classes: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read ecg_info and true_label from the database.

    Each row holds many windows (ecg_info is list[list[float]]) but only one
    true_label, so every window in a row gets that row's label.
    Rows with an unknown label (-1 / NULL) are skipped.

    Returns
    -------
    X : float32 array, shape (num_windows, 180, 1)
    y : int32 array,   shape (num_windows,)
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
    """
    SELECT ecg_info, true_label
    FROM user_details
    WHERE ecg_info IS NOT NULL
      AND true_label IS NOT NULL
      AND true_label >= 0
      AND device_id = ?
    """,
    (EDGE_DEVICE_ID,),
    ).fetchall()
    conn.close()

    X_parts, y_parts = [], []
    for ecg_json, label in rows:
        if n_classes is not None and label >= n_classes:
            print(f"Skipping row with out-of-range label {label}")
            continue

        windows = np.asarray(json.loads(ecg_json), dtype="float32")
        if windows.ndim != 2 or windows.shape[1] != WINDOW_SIZE:
            print(f"Skipping row with unexpected ecg shape {windows.shape}")
            continue

        X_parts.append(windows)
        y_parts.append(np.full(len(windows), label, dtype="int32"))

    if not X_parts:
        return (
            np.empty((0, WINDOW_SIZE, 1), dtype="float32"),
            np.empty((0,), dtype="int32"),
        )

    X = np.concatenate(X_parts, axis=0)[..., np.newaxis]  # (N, 180, 1)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def retrain_last_layer(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    min_samples: int = 32,
    backup: bool = True,
) -> dict:
    """
    Fine-tune the final Dense layer on labeled data from the database and
    save the updated model to oneD_Conv.keras.

    Returns a dict with the final loss/accuracy and number of windows used.
    """
    model = keras.models.load_model(MODEL_PATH)

    head = model.layers[-1]
    if not isinstance(head, keras.layers.Dense):
        raise RuntimeError(f"Expected last layer to be Dense, got {type(head).__name__}")
    n_classes = head.units

    X, y = load_training_data(n_classes=n_classes)
    if len(X) < min_samples:
        raise RuntimeError(
            f"Only {len(X)} labeled windows found (need at least {min_samples}). "
            "Collect more data with true_label set before retraining."
        )

    classes, counts = np.unique(y, return_counts=True)
    print(f"Training on {len(X)} windows. Class counts: {dict(zip(classes.tolist(), counts.tolist()))}")
    if len(classes) < n_classes:
        print(
            f"WARNING: only {len(classes)} of {n_classes} classes present. "
            "The head may become biased toward those classes."
        )

    # Freeze everything except the last layer. Frozen BatchNormalization layers
    # also stay in inference mode, so their running statistics are untouched.
    for layer in model.layers[:-1]:
        layer.trainable = False
    head.trainable = True

    # Must recompile after changing `trainable` for it to take effect.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
    )

    if backup and MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"Backed up previous model to {BACKUP_PATH}")

    model.save(MODEL_PATH)
    print(f"Saved retrained model to {MODEL_PATH}")

    return {
        "num_windows": int(len(X)),
        "loss": float(history.history["loss"][-1]),
        "accuracy": float(history.history["accuracy"][-1]),
    }


if __name__ == "__main__":
    result = retrain_last_layer()
    print(result)