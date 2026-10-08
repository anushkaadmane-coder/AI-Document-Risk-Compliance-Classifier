import sqlite3
import json
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "documents.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a SQLite database connection.
    """

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():
    """
    Create the predictions table if it does not exist.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT NOT NULL,

            risk_level TEXT NOT NULL,

            policy_category TEXT,

            confidence REAL NOT NULL,

            review_required INTEGER NOT NULL DEFAULT 0,

            review_status TEXT NOT NULL,

            important_terms TEXT,

            extracted_text TEXT,

            created_at TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    filename,
    risk_level,
    policy_category,
    confidence,
    review_required,
    review_status,
    important_terms,
    extracted_text
):
    """
    Save a document classification result.
    """

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO predictions (

            filename,
            risk_level,
            policy_category,
            confidence,
            review_required,
            review_status,
            important_terms,
            extracted_text,
            created_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            risk_level,
            policy_category,
            confidence,
            review_required,
            review_status,
            important_terms,
            extracted_text,
            created_at
        )
    )

    prediction_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return prediction_id


# ============================================================
# GET ALL PREDICTIONS
# ============================================================

def get_predictions():
    """
    Return all prediction history.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            risk_level,
            policy_category,
            confidence,
            review_required,
            review_status,
            important_terms,
            created_at
        FROM predictions
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    results = []

    for row in rows:

        item = dict(row)

        # Convert SQLite integer to boolean
        item["review_required"] = bool(
            item["review_required"]
        )

        # Convert JSON string to list
        if item["important_terms"]:

            try:
                item["important_terms"] = json.loads(
                    item["important_terms"]
                )

            except (json.JSONDecodeError, TypeError):

                item["important_terms"] = []

        else:

            item["important_terms"] = []

        results.append(item)

    return results


# ============================================================
# GET SINGLE PREDICTION
# ============================================================

def get_prediction(prediction_id):
    """
    Return one prediction by ID.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            risk_level,
            policy_category,
            confidence,
            review_required,
            review_status,
            important_terms,
            extracted_text,
            created_at
        FROM predictions
        WHERE id = ?
        """,
        (
            prediction_id,
        )
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    result = dict(row)

    result["review_required"] = bool(
        result["review_required"]
    )

    if result["important_terms"]:

        try:

            result["important_terms"] = json.loads(
                result["important_terms"]
            )

        except (json.JSONDecodeError, TypeError):

            result["important_terms"] = []

    else:

        result["important_terms"] = []

    return result


# ============================================================
# GET STATISTICS
# ============================================================

def get_statistics():
    """
    Return dashboard statistics.
    """

    connection = get_connection()

    cursor = connection.cursor()

    # Total documents
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        """
    )

    total_documents = cursor.fetchone()[0]

    # Risk distribution
    cursor.execute(
        """
        SELECT
            risk_level,
            COUNT(*) AS count
        FROM predictions
        GROUP BY risk_level
        """
    )

    risk_rows = cursor.fetchall()

    risk_distribution = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
        "Critical": 0
    }

    for row in risk_rows:

        risk_distribution[
            row["risk_level"]
        ] = row["count"]

    # Manual review count
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE review_required = 1
        """
    )

    manual_review = cursor.fetchone()[0]

    # Average confidence
    cursor.execute(
        """
        SELECT AVG(confidence)
        FROM predictions
        """
    )

    average_confidence = cursor.fetchone()[0]

    connection.close()

    if average_confidence is None:
        average_confidence = 0

    return {
        "total_documents": total_documents,
        "risk_distribution": risk_distribution,
        "manual_review_required": manual_review,
        "average_confidence": round(
            float(average_confidence),
            2
        )
    }


# ============================================================
# CLEAR PREDICTIONS
# ============================================================

def clear_predictions():
    """
    Delete all prediction history.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM predictions
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# INITIALIZE WHEN MODULE IS USED
# ============================================================

initialize_database()