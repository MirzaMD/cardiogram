"""
Real-time ECG + MAX30102 acquisition and preprocessing.

Arduino output format:

    timestamp,ecg,ir,red

Example:

    100412,374,33059,30388

ECG:
    - Acquired at approximately 250 Hz.
    - Windowed into 180 samples.
    - Each window is independently z-normalized.
    - Output shape: (num_windows, 180, 1)

MAX30102:
    - IR and RED are raw PPG measurements.
    - They are NOT SpO2 percentages.
    - Stored alongside the ECG windows.
    - Output shape:
        X_ir  = (num_windows, 180, 1)
        X_red = (num_windows, 180, 1)

No inference is performed here.
"""

import os
import threading
import time

import numpy as np
import serial


# ============================================================
# CONFIG
# ============================================================

SERIAL_PORT = os.environ.get(
    "SERIAL_PORT",
    "/dev/ttyACM0"
)

BAUD_RATE = int(
    os.environ.get(
        "BAUD_RATE",
        115200
    )
)

SERIAL_TIMEOUT = 1.0

MOCK_SERIAL = os.environ.get("MOCK_SERIAL") == "1"

WINDOW_SIZE = 180

WINDOW_STEP = 180

RUN_DURATION_SECONDS = 15


# Only one acquisition may use the serial port at a time.
_ACQ_LOCK = threading.Lock()


# ============================================================
# ECG PREPROCESSING
# ============================================================

def normalize_window(window: np.ndarray) -> np.ndarray:
    """
    Per-window z-normalization.

    Parameters
    ----------
    window:
        Shape (WINDOW_SIZE,)

    Returns
    -------
    np.ndarray
        Normalized ECG window.
    """

    mean = window.mean()
    std = window.std() + 1e-8

    return (window - mean) / std


def prepare_ecg_window(window: np.ndarray) -> np.ndarray:
    """
    Normalize and reshape one ECG window.

    Returns
    -------
    np.ndarray
        Shape: (WINDOW_SIZE, 1)
    """

    normalized = normalize_window(window)

    return normalized.reshape(
        WINDOW_SIZE,
        1
    ).astype(np.float32)


def prepare_ppg_window(window: np.ndarray) -> np.ndarray:
    """
    Prepare raw PPG window.

    No z-normalization is performed here because the
    values are intended to remain raw.

    Returns
    -------
    np.ndarray
        Shape: (WINDOW_SIZE, 1)
    """

    return window.reshape(
        WINDOW_SIZE,
        1
    ).astype(np.float32)


# ============================================================
# SERIAL
# ============================================================

def open_serial_connection(
    port=SERIAL_PORT,
    baud=BAUD_RATE,
    timeout=SERIAL_TIMEOUT
):
    """
    Open Arduino serial connection.
    """

    try:
        ser = serial.Serial(
            port,
            baud,
            timeout=timeout
        )

        # Arduino UNO resets when serial connection opens.
        time.sleep(2)

        # Discard anything generated during reset.
        ser.reset_input_buffer()

        print(
            f"[INFO] Connected to Arduino "
            f"on {port} @ {baud} baud"
        )

        return ser

    except serial.SerialException as e:

        raise RuntimeError(
            f"Could not open serial port {port}. "
            f"Check the Arduino connection and make sure "
            f"no other program is using the port. "
            f"Original error: {e}"
        ) from e


# ============================================================
# SERIAL SAMPLE PARSER
# ============================================================

def read_sample(ser):
    """
    Read one Arduino acquisition line.

    Expected format:

        timestamp,ecg,ir,red

    Example:

        5712448,374,33059,30388

    Returns
    -------
    tuple or None

        (timestamp, ecg, ir, red)

    """

    try:

        line = (
            ser.readline()
            .decode(
                "utf-8",
                errors="ignore"
            )
            .strip()
        )

    except serial.SerialException as e:

        raise RuntimeError(
            "Lost connection to Arduino "
            f"while reading: {e}"
        ) from e

    if not line:
        return None

    # Ignore Arduino header.
    if line.lower() == "timestamp,ecg,ir,red":
        return None

    parts = line.split(",")

    if len(parts) != 4:
        return None

    try:

        timestamp = int(parts[0])

        ecg = float(parts[1])

        ir = float(parts[2])

        red = float(parts[3])

        return (
            timestamp,
            ecg,
            ir,
            red
        )

    except ValueError:

        return None


# ============================================================
# MOCK DATA
# ============================================================

def _mock_samples(
    duration_seconds: float,
    fs: int = 250
):
    """
    Generate synthetic data for UI testing.

    NOT suitable for model evaluation.
    """

    n = int(
        duration_seconds * fs
    )

    t = np.arange(n) / fs

    # Approx. 72 BPM
    beat_period = 60.0 / 72.0

    phase = np.mod(
        t,
        beat_period
    )

    # Synthetic ECG-like spike
    qrs = np.exp(
        -(
            (phase - 0.2) ** 2
        )
        /
        (2 * 0.01 ** 2)
    )

    ecg = (
        512
        + 300 * qrs
        + np.random.normal(
            0,
            8,
            n
        )
    )

    # Synthetic PPG
    ppg_wave = (
        1000
        * np.exp(
            -(
                (phase - 0.25) ** 2
            )
            /
            (2 * 0.04 ** 2)
        )
    )

    ir = (
        33000
        + ppg_wave
        + np.random.normal(
            0,
            50,
            n
        )
    )

    red = (
        30000
        + 0.8 * ppg_wave
        + np.random.normal(
            0,
            40,
            n
        )
    )

    timestamps = (
        np.arange(n)
        * 4000
    )

    return (
        timestamps.tolist(),
        ecg.tolist(),
        ir.tolist(),
        red.tolist()
    )


# ============================================================
# MAIN ACQUISITION
# ============================================================

def acquire_and_preprocess(
    duration_seconds: float = RUN_DURATION_SECONDS
):
    """
    Acquire ECG + MAX30102 data and preprocess ECG.

    Returns
    -------
    X_ecg:
        Shape:
            (num_windows, 180, 1)

        Per-window z-normalized ECG.

    X_ir:
        Shape:
            (num_windows, 180, 1)

        Raw MAX30102 IR values.

    X_red:
        Shape:
            (num_windows, 180, 1)

        Raw MAX30102 RED values.

    timestamps:
        Shape:
            (num_samples,)

        Arduino timestamps in microseconds.
    """

    if not _ACQ_LOCK.acquire(
        blocking=False
    ):

        raise RuntimeError(
            "A capture is already in progress."
        )

    try:

        timestamps = []
        ecg_samples = []
        ir_samples = []
        red_samples = []

        # ----------------------------------------------------
        # MOCK MODE
        # ----------------------------------------------------

        if MOCK_SERIAL:

            print(
                "[INFO] MOCK_SERIAL=1: "
                f"generating synthetic data for "
                f"{duration_seconds}s..."
            )

            (
                timestamps,
                ecg_samples,
                ir_samples,
                red_samples
            ) = _mock_samples(
                duration_seconds
            )

        # ----------------------------------------------------
        # REAL ARDUINO
        # ----------------------------------------------------

        else:

            ser = open_serial_connection()

            start_time = time.monotonic()

            print(
                "[INFO] Acquiring ECG + "
                f"MAX30102 for "
                f"{duration_seconds}s..."
            )

            try:

                while (
                    time.monotonic()
                    - start_time
                ) < duration_seconds:

                    sample = read_sample(ser)

                    if sample is None:
                        continue

                    (
                        timestamp,
                        ecg,
                        ir,
                        red
                    ) = sample

                    timestamps.append(
                        timestamp
                    )

                    ecg_samples.append(
                        ecg
                    )

                    ir_samples.append(
                        ir
                    )

                    red_samples.append(
                        red
                    )

            finally:

                ser.close()

                print(
                    "[INFO] Serial connection closed."
                )

    finally:

        _ACQ_LOCK.release()

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    ecg = np.asarray(
        ecg_samples,
        dtype=np.float64
    )

    ir = np.asarray(
        ir_samples,
        dtype=np.float64
    )

    red = np.asarray(
        red_samples,
        dtype=np.float64
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.int64
    )

    print(
        f"[INFO] Collected "
        f"{len(ecg)} samples."
    )

    # ========================================================
    # NOT ENOUGH DATA
    # ========================================================

    empty = np.empty(
        (
            0,
            WINDOW_SIZE,
            1
        ),
        dtype=np.float32
    )

    if len(ecg) < WINDOW_SIZE:

        print(
            "[WARN] Fewer than "
            f"{WINDOW_SIZE} ECG samples "
            "were collected."
        )

        return (
            empty,
            empty.copy(),
            empty.copy(),
            timestamps
        )

    # ========================================================
    # WINDOWING
    # ========================================================

    ecg_windows = []
    ir_windows = []
    red_windows = []

    for start in range(
        0,
        len(ecg) - WINDOW_SIZE + 1,
        WINDOW_STEP
    ):

        end = (
            start
            + WINDOW_SIZE
        )

        # ECG
        ecg_window = ecg[
            start:end
        ]

        ecg_windows.append(
            prepare_ecg_window(
                ecg_window
            )
        )

        # IR
        ir_window = ir[
            start:end
        ]

        ir_windows.append(
            prepare_ppg_window(
                ir_window
            )
        )

        # RED
        red_window = red[
            start:end
        ]

        red_windows.append(
            prepare_ppg_window(
                red_window
            )
        )

    # ========================================================
    # STACK
    # ========================================================

    X_ecg = np.stack(
        ecg_windows,
        axis=0
    )

    X_ir = np.stack(
        ir_windows,
        axis=0
    )

    X_red = np.stack(
        red_windows,
        axis=0
    )

    print(
        "[INFO] Acquisition complete."
    )

    print(
        f"[INFO] ECG shape: "
        f"{X_ecg.shape}"
    )

    print(
        f"[INFO] IR shape: "
        f"{X_ir.shape}"
    )

    print(
        f"[INFO] RED shape: "
        f"{X_red.shape}"
    )

    return (
        X_ecg,
        X_ir,
        X_red,
        timestamps
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    (
        ecg_info,
        ir_info,
        red_info,
        timestamps
    ) = acquire_and_preprocess()

    print(
        f"ECG: {ecg_info.shape}"
    )

    print(
        f"IR: {ir_info.shape}"
    )

    print(
        f"RED: {red_info.shape}"
    )

    print(
        f"Timestamps: {timestamps.shape}"
    )