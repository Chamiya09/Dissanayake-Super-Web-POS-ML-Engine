from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].astype(np.int32)
    return out


def load_feature_data(path: Path, date_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = {date_col, "ProductID", "ProductName", "Category", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {path.name}: {sorted(missing)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    df = df.dropna(subset=[date_col, "ProductID", "ProductName", "Category", "Quantity"]).copy()
    df = optimize_dtypes(df)
    df = df.sort_values(["ProductID", date_col]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows found after load/cleanup: {path}")

    return df


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "ProductName",
        "Category",
        "Quantity",
        "log_quantity",
        "Week_Ending_Sunday",
        "Month_Start",
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def time_based_train_test_split(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[(df[date_col] >= "2019-01-01") & (df[date_col] < "2025-01-01")].copy()
    test_df = df[(df[date_col] >= "2025-01-01") & (df[date_col] < "2026-01-01")].copy()

    if train_df.empty:
        raise ValueError("Training split is empty for 2019-2024.")
    if test_df.empty:
        raise ValueError("Test split is empty for 2025.")

    return train_df, test_df


def prepare_model_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df["Quantity"].to_numpy(dtype=np.float32)
    y_test = test_df["Quantity"].to_numpy(dtype=np.float32)

    return X_train, X_test, y_train, y_test
