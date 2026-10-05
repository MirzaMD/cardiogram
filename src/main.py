import os
import time

import numpy as np
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utilities.extrt_pred import (
    extract_preprocess_predict_and_save
)

from db.database import (
    get_users_detail,
    update_true_label
)

from utilities.retrain import (
    retrain_last_layer
)

from utilities.fed_weights import (
    get_para,
    update_para
)

from utilities.client_model import (
    reload_model
)


# ============================================================
# CONFIGURATION
# ============================================================

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

SERVER_URL = os.environ.get(
    "SERVER_URL",
    "http://localhost:8000"
).rstrip("/")

API_KEY = os.environ.get(
    "API_KEY",
    "default_api_key"
)

FRONTEND_ORIGIN = os.environ.get(
    "http://localhost:3000",
    "*"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LifeSenseAI",
    description=(
        "Edge API for real-time ECG arrhythmia detection "
        "and MAX30102 PPG acquisition."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        FRONTEND_ORIGIN,
        "*",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def read_root():
    """
    Basic API health check.
    """

    return {
        "message": "Welcome to the LifeSenseAI API",
        "device_id": EDGE_DEVICE_ID,
        "device": EDGE_DEVICE_NAME,
        "status": "online",
    }


# ============================================================
# ECG + PPG ACQUISITION AND INFERENCE
# ============================================================

@app.post("/predict")
def predict(
    duration_seconds: float = 15
):
    """
    Acquire ECG + MAX30102 data, preprocess the ECG,
    run inference and save the result to SQLite.

    Parameters
    ----------
    duration_seconds:
        Duration of Arduino acquisition.
    """

    if duration_seconds <= 0:
        raise HTTPException(
            status_code=400,
            detail="duration_seconds must be greater than 0."
        )

    if duration_seconds > 300:
        raise HTTPException(
            status_code=400,
            detail="Maximum acquisition duration is 300 seconds."
        )

    try:

        result = (
            extract_preprocess_predict_and_save(
                duration_seconds=duration_seconds,

                device_id=EDGE_DEVICE_ID,

                device=EDGE_DEVICE_NAME,
            )
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction pipeline failed: {e}"
        ) from e

    return {
        "id": result["id"],

        "device_id": result["device_id"],

        "device": result["device"],

        "prediction": result["prediction"],

        "prediction_label":
            result["prediction_label"],

        "true_label":
            result["true_label"],

        "ecg_info":
            result["ecg_info"],

        # Raw MAX30102 PPG
        "ir_info":
            result["ir_info"],

        "red_info":
            result["red_info"],

        # Structured IR + RED representation
        "oxygen_info":
            result["oxygen_info"],

        "timestamps":
            result["timestamps"],
    }


# ============================================================
# USER / DEVICE DATA
# ============================================================

@app.get("/data")
def user_info():
    """
    Return all stored records for this edge device.
    """

    user_details = get_users_detail(
        device_id=EDGE_DEVICE_ID
    )

    if not user_details:

        return {
            "message": (
                "No data found for the given "
                "device_id."
            )
        }

    return user_details


# ============================================================
# UPDATE TRUE LABEL
# ============================================================

@app.put("/true-label-update/{id}")
def update_label(
    id: int,
    true_label: int
):
    """
    Update the ground-truth label of an existing
    database record.

    Example:

        PUT /true-label-update/12?true_label=3
    """

    if true_label not in (-1, 0, 1, 2, 3, 4):

        raise HTTPException(
            status_code=400,
            detail=(
                "true_label must be -1 or one of "
                "0, 1, 2, 3, 4."
            )
        )

    updated = update_true_label(
        id=id,
        true_label=true_label
    )

    if updated == 0:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No record found with id {id}."
            )
        )

    return {
        "message": (
            f"True label for record with id "
            f"{id} updated to {true_label}."
        ),

        "id": id,

        "true_label": true_label,
    }


# ============================================================
# LOCAL TRAINING
# ============================================================

@app.post("/local-train")
def local_train():
    """
    Retrain the local model's final layer.
    """

    try:
        
        result = retrain_last_layer()

        # Reload the newly trained model.
        reload_model()

    except RuntimeError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Local training failed: {e}"
        ) from e

    return {
        "message": "Local training completed.",

        **result
    }


# ============================================================
# FEDERATED WEIGHT AGGREGATION
# ============================================================

@app.post("/aggregate-weights")
def aggregate_weights(
    timeout: int = 120,
    poll: int = 5
):
    """
    Upload local model weights to the central FedAvg server.

    If the server is waiting for other edge devices,
    poll until the global model for the next round
    becomes available.
    """

    if timeout <= 0:
        raise HTTPException(
            status_code=400,
            detail="timeout must be greater than 0."
        )

    if poll <= 0:
        raise HTTPException(
            status_code=400,
            detail="poll must be greater than 0."
        )

    headers = {
        "x-api-key": API_KEY
    }

    # --------------------------------------------------------
    # Get local model parameters
    # --------------------------------------------------------

    local_weights = get_para()

    payload = {
        "device_id": str(
            EDGE_DEVICE_ID
        ),

        "weights": [
            weight.tolist()
            for weight in local_weights
        ],
    }

    # --------------------------------------------------------
    # Upload local weights
    # --------------------------------------------------------

    try:

        response = requests.post(
            f"{SERVER_URL}/fedavg",

            json=payload,

            headers=headers,

            timeout=30,
        )

        response.raise_for_status()

        body = response.json()

    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not communicate with "
                f"federated server: {e}"
            )
        ) from e

    # --------------------------------------------------------
    # Server says it is waiting for other clients
    # --------------------------------------------------------

    if body.get("status") == "waiting":

        start_round = body.get(
            "round",
            0
        )

        deadline = (
            time.time()
            + timeout
        )

        while time.time() < deadline:

            time.sleep(poll)

            try:

                global_response = requests.get(
                    f"{SERVER_URL}/global_weights",

                    headers=headers,

                    timeout=30,
                )

            except requests.RequestException:

                continue

            if not global_response.ok:
                continue

            try:

                global_body = (
                    global_response.json()
                )

            except ValueError:

                continue

            if (
                global_body.get("round", 0)
                > start_round
            ):

                body = global_body

                break

        else:

            raise HTTPException(
                status_code=504,
                detail=(
                    "Global model was not ready "
                    "within the requested timeout."
                )
            )

    # --------------------------------------------------------
    # Validate global weights
    # --------------------------------------------------------

    if "weights" not in body:

        raise HTTPException(
            status_code=502,
            detail=(
                "Federated server response did not "
                "contain global weights."
            )
        )

    try:

        weights_as_arrays = [
            np.asarray(
                weight,
                dtype=np.float32
            )
            for weight in body["weights"]
        ]

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Invalid global weights received: {e}"
            )
        ) from e

    # --------------------------------------------------------
    # Apply global parameters
    # --------------------------------------------------------

    try:

        update_para(
            weights_as_arrays
        )

        # update_para() already reloads the model
        # according to your current implementation.

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not apply global weights: {e}"
            )
        ) from e

    return {
        "message": "Global weights applied.",

        "round":
            body.get("round"),

        "device_id":
            EDGE_DEVICE_ID,
    }
