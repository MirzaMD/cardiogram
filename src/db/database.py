"""
SQLite database for LifeSenseAI edge-device readings.

Stored data:

    device_id
    device
    ecg_info
    prediction
    true_label
    oxygen_info

ECG format
----------
ecg_info:

    list[list[float]]

    [
        [180 ECG samples],
        [180 ECG samples],
        ...
    ]


MAX30102 format
---------------
oxygen_info:

    list[list[list[float]]]

    [
        [
            [IR, RED],
            [IR, RED],
            ...
        ],
        [
            [IR, RED],
            [IR, RED],
            ...
        ]
    ]

Each outer element corresponds to one ECG window.

SQLite stores the nested lists as JSON TEXT because SQLite
does not have a native array/list type.

Dependencies
------------
Standard library only:

    sqlite3
    json
    pathlib
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


# ============================================================
# DATABASE LOCATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DB_PATH = (
    PROJECT_ROOT
    / "src"
    / "user_details.db"
)


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database(
    db_path: str | Path = DB_PATH
) -> None:
    """
    Create the SQLite database and user_details table
    if they do not already exist.
    """

    db_path = Path(db_path)

    # Ensure parent directory exists.
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        db_path
    )

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_details (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                device_id INTEGER,

                device TEXT NOT NULL,

                ecg_info TEXT,

                prediction INTEGER,

                true_label INTEGER,

                oxygen_info TEXT

            )
            """
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# INSERT
# ============================================================

def insert_user_details(
    device_id: int,
    device: str,
    ecg_info: list[list[float]],
    prediction: int,
    true_label: int,
    oxygen_info: list[list[list[float]]],
    db_path: str | Path = DB_PATH,
) -> int:
    """
    Insert one UserDetails record.

    Parameters
    ----------
    device_id:
        Numeric edge-device identifier.

    device:
        Human-readable edge-device name.

    ecg_info:
        ECG windows.

        Shape conceptually:

            (num_windows, 180)

    prediction:
        Predicted ECG class index.

    true_label:
        Ground-truth class index.
        Use -1 when unknown.

    oxygen_info:
        Raw MAX30102 PPG.

        Structure:

            [
                [
                    [IR, RED],
                    [IR, RED],
                    ...
                ],
                ...
            ]

    Returns
    -------
    int
        SQLite row ID of inserted record.
    """

    # Make sure the database exists.
    create_database(
        db_path
    )

    conn = sqlite3.connect(
        db_path
    )

    try:

        cur = conn.execute(
            """
            INSERT INTO user_details
            (
                device_id,
                device,
                ecg_info,
                prediction,
                true_label,
                oxygen_info
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                device_id,
                device,
                json.dumps(ecg_info),
                prediction,
                true_label,
                json.dumps(oxygen_info),
            ),
        )

        conn.commit()

        return int(
            cur.lastrowid
        )

    finally:

        conn.close()


# ============================================================
# READ
# ============================================================

def get_users_detail(
    device_id: int,
    db_path: str | Path = DB_PATH,
) -> list[dict]:
    """
    Fetch all records belonging to a device.

    JSON fields are decoded back into Python lists.
    """

    conn = sqlite3.connect(
        db_path
    )

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                device_id,
                device,
                ecg_info,
                prediction,
                true_label,
                oxygen_info
            FROM user_details
            WHERE device_id = ?
            """,
            (device_id,),
        )

        rows = cursor.fetchall()

    finally:

        conn.close()

    if not rows:
        return []

    return [
        {
            "id": row[0],

            "device_id": row[1],

            "device": row[2],

            "ecg_info":
                json.loads(row[3])
                if row[3]
                else [],

            "prediction": row[4],

            "true_label": row[5],

            "oxygen_info":
                json.loads(row[6])
                if row[6]
                else [],
        }
        for row in rows
    ]


# ============================================================
# UPDATE TRUE LABEL
# ============================================================

def update_true_label(
    id: int,
    true_label: int,
    db_path: str | Path = DB_PATH,
) -> int:
    """
    Update the ground-truth label for an existing record.

    Returns
    -------
    int
        Number of rows updated.
    """

    conn = sqlite3.connect(
        db_path
    )

    try:

        cur = conn.execute(
            """
            UPDATE user_details
            SET true_label = ?
            WHERE id = ?
            """,
            (
                true_label,
                id,
            ),
        )

        conn.commit()

        return cur.rowcount

    finally:

        conn.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_database()

    print(
        f"[INFO] Database ready at: {DB_PATH}"
    )

