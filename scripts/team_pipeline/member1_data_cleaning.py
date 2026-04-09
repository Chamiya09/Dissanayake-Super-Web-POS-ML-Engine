from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_FALLBACK_NAME = "DISSANAYAKA_POS_DATASET_2018-2025.csv"


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def resolve_input_with_fallback(input_path: Path) -> Path:
    if input_path.exists():
        return input_path

    fallback_candidates = [
        input_path.parent / RAW_FALLBACK_NAME,
        input_path.parent / "DISSANAYAKA_POS_DATASET_2019-2025.csv",
        input_path.parent / "DISSANAYAKA_POS_DATASET_2018-2025.csv",
    ]

    for fallback in fallback_candidates:
        if fallback.exists():
            print(f"[WARN] Input not found at {input_path}. Using fallback: {fallback}")
            return fallback

    discovered = sorted(input_path.parent.glob("DISSANAYAKA_POS_DATASET_*.csv"))
    if discovered:
        selected = discovered[-1]
        print(f"[WARN] Input not found at {input_path}. Using discovered dataset: {selected}")
        return selected

    raise FileNotFoundError(f"Dataset not found: {input_path}")


def round_to_nearest_10(series: pd.Series) -> pd.Series:
    return (series.div(10).round().mul(10)).astype("int64")


def align_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Minimal compatibility mapping for minor source naming differences.
    rename_map = {
        "Payment Method": "Payment_Method",
        "Pricing Unit": "PricingUnit",
        "Transaction ID": "TransactionID",
    }
    existing = {k: v for k, v in rename_map.items() if k in out.columns and v not in out.columns}
    if existing:
        out = out.rename(columns=existing)

    return out


def load_and_validate(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = align_input_schema(df)

    required_cols = {
        "TransactionID",
        "Date",
        "Time",
        "ProductID",
        "ProductName",
        "Category",
        "UnitPrice",
        "BuyingPrice",
        "SellingPrice",
        "Quantity",
        "Total_LKR",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Keep the same pipeline behavior while accepting minor time format variations.
    time_str = df["Time"].astype(str).str.strip()
    parsed_time = pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")
    fallback_mask = parsed_time.isna()
    if fallback_mask.any():
        parsed_time.loc[fallback_mask] = pd.to_datetime(time_str[fallback_mask], errors="coerce")
    df["Time"] = parsed_time.dt.time

    numeric_cols = ["UnitPrice", "BuyingPrice", "SellingPrice", "Quantity", "Total_LKR"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "TransactionID",
            "Date",
            "Time",
            "ProductID",
            "ProductName",
            "Category",
            "UnitPrice",
            "BuyingPrice",
            "SellingPrice",
            "Quantity",
            "Total_LKR",
        ]
    ).copy()

    if df.empty:
        raise ValueError("No rows available after null and type validation.")

    datetime_index = pd.to_datetime(
        df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
        errors="coerce",
    )
    if datetime_index.isna().any():
        raise ValueError("Failed to construct valid transaction datetime for all rows.")

    df["TransactionDateTime"] = datetime_index

    # Ensure transaction records are chronologically ordered by Date + Time + TransactionID.
    df = df.sort_values(["Date", "Time", "TransactionID"]).reset_index(drop=True)

    return df


def apply_price_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price_cols = ["UnitPrice", "BuyingPrice", "SellingPrice", "Total_LKR"]

    for col in price_cols:
        out[col] = round_to_nearest_10(out[col])

    out["Quantity"] = out["Quantity"].astype("int64")

    return out
