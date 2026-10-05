import os

import numpy as np

from db.database import create_database, insert_user_details
from utilities.preprocess import acquire_and_preprocess
from utilities.client_model import get_model


# ------------------------------------------------------------
# Label mapping
# Must match the mapping used during model training.
# ------------------------------------------------------------

LABELS = {
    0: "F",
    1: "N",
    2: "Q",
    3: "S",
    4: "V",
}


# ------------------------------------------------------------
# Edge-device configuration
# ------------------------------------------------------------

EDGE_DEVICE_ID = int(
    os.environ.get(
        "EDGE_DEVICE_ID",
        1
    )
)

EDGE_DEVICE_NAME = os.environ.get(
    "EDGE_DEVICE_NAME",
    "New Edge"
)


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def extract_preprocess_predict_and_save(
    duration_seconds: float = 15,
    true_label: int = -1,
    device_id: int = EDGE_DEVICE_ID,
    device: str = EDGE_DEVICE_NAME,
) -> dict:
    """
    Runs one complete acquisition → preprocessing →
    inference → database-storage cycle.

    Pipeline
    --------
    1. Acquire ECG + MAX30102 raw data from Arduino.
    2. Preprocess ECG into 180-sample z-normalized windows.
    3. Keep MAX30102 IR and RED PPG data aligned with
       the same windows.
    4. Run the trained ECG model.
    5. Save the result to the database.
    6. Return the saved record.

    Parameters
    ----------
    duration_seconds:
        Duration of Arduino acquisition.

    true_label:
        Ground-truth class index if known.
        Use -1 for live/unknown data.

    device_id:
        Numeric identifier of the edge device.

    device:
        Human-readable edge-device name.

    Returns
    -------
    dict
        Contains device information, ECG data, PPG data,
        prediction and database record ID.
    """

    # ========================================================
    # 1 + 2. ACQUIRE AND PREPROCESS
    # ========================================================

    (
        X_ecg,
        X_ir,
        X_red,
        timestamps,
    ) = acquire_and_preprocess(
        duration_seconds=duration_seconds
    )

    # X_ecg:
    #     (num_windows, 180, 1)
    #
    # X_ir:
    #     (num_windows, 180, 1)
    #
    # X_red:
    #     (num_windows, 180, 1)

    if X_ecg.shape[0] == 0:
        raise RuntimeError(
            "No ECG windows were acquired. "
            "Check the Arduino connection, serial port, "
            "and acquisition duration."
        )

    # ========================================================
    # 3. MODEL INFERENCE
    # ========================================================

    model = get_model()

    predicted_proba = model.predict(
        X_ecg,
        verbose=0
    )

    # One prediction per ECG window.
    predicted_indices = np.argmax(
        predicted_proba,
        axis=1
    )

    # --------------------------------------------------------
    # For now, use the first window as the recording-level
    # prediction, preserving your previous behaviour.
    # --------------------------------------------------------

    prediction = int(
        predicted_indices[0]
    )

    prediction_label = LABELS.get(
        prediction,
        "UNKNOWN"
    )

    # ========================================================
    # 4. PREPARE DATA FOR DATABASE
    # ========================================================

    # Remove the trailing CNN channel dimension.
    #
    # Before:
    #   (num_windows, 180, 1)
    #
    # After:
    #   (num_windows, 180)

    ecg_info = X_ecg.squeeze(
        -1
    ).tolist()

    # --------------------------------------------------------
    # MAX30102 data
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # X_ir and X_red are RAW PPG measurements.
    # They are NOT SpO2 percentages.
    #
    # We keep both streams separately here.
    # --------------------------------------------------------

    ir_info = X_ir.squeeze(
        -1
    ).tolist()

    red_info = X_red.squeeze(
        -1
    ).tolist()

    # Arduino timestamps are retained separately.
    timestamps_info = timestamps.tolist()

    # --------------------------------------------------------
    # Current database schema compatibility
    # --------------------------------------------------------
    #
    # If your existing database only has "oxygen_info",
    # store IR + RED together as:
    #
    # [
    #     {
    #         "ir": [...],
    #         "red": [...]
    #     },
    #     ...
    # ]
    #
    # If oxygen_info is currently expected to be a
    # list[list[float]], you may instead store IR temporarily.
    #
    # Here we use a structured representation because it
    # prevents losing the RED channel.
    # --------------------------------------------------------

    oxygen_info = [
        {
            "ir": ir_window,
            "red": red_window,
        }
        for ir_window, red_window
        in zip(
            ir_info,
            red_info
        )
    ]

    # ========================================================
    # 5. SAVE TO DATABASE
    # ========================================================

    create_database()

    record_id = insert_user_details(
        device_id=device_id,
        device=device,
        ecg_info=ecg_info,
        prediction=prediction,
        true_label=true_label,
        oxygen_info=oxygen_info,
    )

    # ========================================================
    # 6. RETURN RESULT
    # ========================================================

    return {
        "device_id": device_id,
        "device": device,

        "ecg_info": ecg_info,

        "prediction": prediction,
        "prediction_label": prediction_label,

        "true_label": true_label,

        # Raw MAX30102 PPG
        "oxygen_info": oxygen_info,

        # Optional additional information
        "ir_info": ir_info,
        "red_info": red_info,
        "timestamps": timestamps_info,

        "id": record_id,
    }


# ============================================================
# Optional direct test
# ============================================================

if __name__ == "__main__":

    result = extract_preprocess_predict_and_save(
        duration_seconds=15
    )

    print(
        "[INFO] Pipeline complete."
    )

    print(
        f"Prediction: "
        f"{result['prediction_label']}"
    )

    print(
        f"Record ID: "
        f"{result['id']}"
    )
