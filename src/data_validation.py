"""Validation of delivery data (generated or uploaded as CSV).

``validate_dataframe`` returns a list of human-readable problems.
``load_and_validate_csv`` raises ``DataValidationError`` with all problems listed.
"""

import numpy as np
import pandas as pd

from .data_generator import COLUMNS

REQUIRED_COLUMNS = [c for c in COLUMNS if c != "package_size"]   # package_size is optional
COORDINATE_LIMIT = 10_000.0
MAX_DELIVERIES = 50


class DataValidationError(ValueError):
    """Raised when a delivery dataset is not valid."""


def _to_bool(value):
    """Convert True/False/1/0/'yes'/'true' ... to bool; return None if unrecognised."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "1.0"):
        return True
    if text in ("false", "0", "no", "n", "0.0"):
        return False
    return None


def validate_dataframe(df):
    """Check the dataset and return a list of error messages (empty list = valid)."""
    problems = []

    if df is None or len(df) == 0:
        return ["The dataset is empty."]

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        return [f"Missing required column(s): {', '.join(missing_columns)}."]

    # --- missing values in required columns -------------------------------------
    for column in REQUIRED_COLUMNS:
        n_missing = int(df[column].isna().sum())
        if n_missing:
            problems.append(f"Column '{column}' has {n_missing} missing value(s).")
    if problems:
        return problems    # later checks would only produce confusing follow-up errors

    # --- duplicate IDs ---------------------------------------------------------
    duplicated_ids = df.loc[df["delivery_id"].duplicated(keep=False), "delivery_id"].unique()
    if len(duplicated_ids):
        problems.append(f"Duplicate delivery_id value(s): {', '.join(map(str, duplicated_ids))}.")

    # --- numeric columns ---------------------------------------------------------
    numeric = {}
    for column in ["x_coordinate", "y_coordinate", "priority",
                   "service_time_minutes", "delivery_deadline_minutes"]:
        converted = pd.to_numeric(df[column], errors="coerce")
        n_bad = int(converted.isna().sum())
        if n_bad:
            problems.append(f"Column '{column}' has {n_bad} non-numeric value(s).")
        numeric[column] = converted
    if problems:
        return problems

    # --- depot flag --------------------------------------------------------------
    depot_flags = df["is_depot"].map(_to_bool)
    if depot_flags.isna().any():
        problems.append("Column 'is_depot' must contain True/False (or 1/0) values.")
        return problems
    n_depots = int(depot_flags.sum())
    if n_depots != 1:
        problems.append(f"Exactly one depot row (is_depot = True) is required; found {n_depots}.")
    n_customers = len(df) - n_depots
    if n_customers < 1:
        problems.append("At least one delivery (non-depot) row is required.")
    if n_customers > MAX_DELIVERIES:
        problems.append(f"At most {MAX_DELIVERIES} deliveries are supported; found {n_customers}.")

    # --- coordinates -------------------------------------------------------------
    for column in ["x_coordinate", "y_coordinate"]:
        values = numeric[column]
        if not np.isfinite(values).all():
            problems.append(f"Column '{column}' contains infinite values.")
        elif (values.abs() > COORDINATE_LIMIT).any():
            problems.append(f"Column '{column}' has values outside [-{COORDINATE_LIMIT:g}, {COORDINATE_LIMIT:g}].")
    points = df[["x_coordinate", "y_coordinate"]].astype(float)
    duplicated_points = points.duplicated(keep=False) & ~depot_flags.values
    if duplicated_points.any():
        problems.append("Two or more locations share exactly the same coordinates.")

    # --- priority ----------------------------------------------------------------
    priority = numeric["priority"]
    customers = ~depot_flags
    bad_priority = customers & ((priority < 1) | (priority > 5) | (priority != priority.round()))
    if bad_priority.any():
        problems.append("Priority must be a whole number from 1 to 5 "
                        f"(invalid in row(s): {', '.join(map(str, df.index[bad_priority] + 2))}).")

    # --- times -------------------------------------------------------------------
    if (customers & (numeric["service_time_minutes"] < 0)).any():
        problems.append("service_time_minutes must not be negative.")
    if (customers & (numeric["delivery_deadline_minutes"] <= 0)).any():
        problems.append("delivery_deadline_minutes must be positive for every delivery.")

    return problems


def clean_dataframe(df):
    """Return a validated, cleaned copy with the depot as the first row.

    Raises DataValidationError when the data is invalid.
    """
    problems = validate_dataframe(df)
    if problems:
        raise DataValidationError("Invalid delivery data:\n- " + "\n- ".join(problems))

    cleaned = df.copy()
    cleaned["is_depot"] = cleaned["is_depot"].map(_to_bool)
    for column in ["x_coordinate", "y_coordinate"]:
        cleaned[column] = cleaned[column].astype(float)
    for column in ["priority", "service_time_minutes", "delivery_deadline_minutes"]:
        cleaned[column] = pd.to_numeric(cleaned[column]).round().astype(int)
    if "package_size" not in cleaned.columns:
        cleaned["package_size"] = "Unknown"
    cleaned["location_name"] = cleaned["location_name"].astype(str)

    # Put the depot first so that it is always node 0 in the algorithms.
    cleaned = cleaned.sort_values("is_depot", ascending=False, kind="stable").reset_index(drop=True)
    return cleaned[["delivery_id", "location_name", "x_coordinate", "y_coordinate", "priority",
                    "service_time_minutes", "delivery_deadline_minutes", "package_size", "is_depot"]]


def load_and_validate_csv(source):
    """Read a CSV (path or file-like object), validate it and return the cleaned DataFrame."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise DataValidationError("The CSV file is empty.")
    except pd.errors.ParserError as error:
        raise DataValidationError(f"The file could not be parsed as CSV: {error}")
    except UnicodeDecodeError:
        raise DataValidationError("The file is not a text CSV file (encoding error).")
    return clean_dataframe(df)
