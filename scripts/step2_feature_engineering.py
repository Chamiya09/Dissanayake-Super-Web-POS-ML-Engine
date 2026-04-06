from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Advanced feature engineering for weekly and monthly aggregations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cleaned_pos_data.csv"),
        help="Path to cleaned daily dataset",
    )
    parser.add_argument(
        "--weekly-output",
        type=Path,
        default=Path("data/processed/final_weekly_features.csv"),
        help="Output path for weekly feature dataset",
    )
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=Path("data/processed/final_monthly_features.csv"),
        help="Output path for monthly feature dataset",
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def load_cleaned_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = {
        "Date",
        "ProductID",
        "ProductName",
        "Category",
        "Quantity",
        "BuyingPrice",
        "SellingPrice",
        "UnitPrice",
        "Total_LKR",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Quantity", "BuyingPrice", "SellingPrice", "UnitPrice", "Total_LKR"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "ProductID", "ProductName", "Category", *numeric_cols]).copy()
    df = df.sort_values(["ProductID", "Date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows available after loading and validation.")

    return df


def add_common_flags(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out["Is_Avurudu_Month"] = (out[date_col].dt.month == 4).astype(np.int8)
    out["Is_Vesak_Month"] = (out[date_col].dt.month == 5).astype(np.int8)
    out["Is_Christmas_Month"] = (out[date_col].dt.month == 12).astype(np.int8)
    return out


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(
            [
                "ProductID",
                "ProductName",
                "Category",
                pd.Grouper(key="Date", freq="W-SUN"),
            ],
            as_index=False,
        )
        .agg(
            Quantity=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            BuyingPrice=("BuyingPrice", "mean"),
            SellingPrice=("SellingPrice", "mean"),
            Total_LKR=("Total_LKR", "sum"),
        )
        .rename(columns={"Date": "Week_Ending_Sunday"})
        .sort_values(["ProductID", "Week_Ending_Sunday"])
        .reset_index(drop=True)
    )

    g = weekly.groupby("ProductID")
    weekly["lag_1_week"] = g["Quantity"].shift(1)
    weekly["lag_2_weeks"] = g["Quantity"].shift(2)
    weekly["rolling_mean_4_weeks"] = g["Quantity"].shift(1).rolling(window=4, min_periods=4).mean()

    weekly["price_difference"] = weekly["SellingPrice"] - weekly["BuyingPrice"]
    weekly = add_common_flags(weekly, "Week_Ending_Sunday")

    # Approximation: a week is payday week if its ending Sunday falls in the last 7 days of month.
    days_in_month = weekly["Week_Ending_Sunday"].dt.days_in_month
    weekly["Is_Payday_Week"] = (weekly["Week_Ending_Sunday"].dt.day >= (days_in_month - 6)).astype(np.int8)

    weekly = weekly.dropna(subset=["lag_1_week", "lag_2_weeks", "rolling_mean_4_weeks"]).copy()
    weekly["log_quantity"] = np.log1p(weekly["Quantity"].clip(lower=0))

    float_cols = [
        "Quantity",
        "UnitPrice",
        "BuyingPrice",
        "SellingPrice",
        "Total_LKR",
        "lag_1_week",
        "lag_2_weeks",
        "rolling_mean_4_weeks",
        "price_difference",
        "log_quantity",
    ]
    weekly[float_cols] = weekly[float_cols].astype(np.float32)

    return weekly


def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(
            [
                "ProductID",
                "ProductName",
                "Category",
                pd.Grouper(key="Date", freq="MS"),
            ],
            as_index=False,
        )
        .agg(
            Quantity=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            BuyingPrice=("BuyingPrice", "mean"),
            SellingPrice=("SellingPrice", "mean"),
            Total_LKR=("Total_LKR", "sum"),
        )
        .rename(columns={"Date": "Month_Start"})
        .sort_values(["ProductID", "Month_Start"])
        .reset_index(drop=True)
    )

    g = monthly.groupby("ProductID")
    monthly["lag_1_month"] = g["Quantity"].shift(1)
    monthly["lag_2_months"] = g["Quantity"].shift(2)
    monthly["rolling_mean_3_months"] = g["Quantity"].shift(1).rolling(window=3, min_periods=3).mean()

    monthly["price_difference"] = monthly["SellingPrice"] - monthly["BuyingPrice"]
    monthly = add_common_flags(monthly, "Month_Start")

    monthly = monthly.dropna(subset=["lag_1_month", "lag_2_months", "rolling_mean_3_months"]).copy()
    monthly["log_quantity"] = np.log1p(monthly["Quantity"].clip(lower=0))

    float_cols = [
        "Quantity",
        "UnitPrice",
        "BuyingPrice",
        "SellingPrice",
        "Total_LKR",
        "lag_1_month",
        "lag_2_months",
        "rolling_mean_3_months",
        "price_difference",
        "log_quantity",
    ]
    monthly[float_cols] = monthly[float_cols].astype(np.float32)

    return monthly


def run(input_path: Path, weekly_output: Path, monthly_output: Path) -> None:
    df = load_cleaned_data(input_path)

    weekly = build_weekly_features(df)
    monthly = build_monthly_features(df)

    weekly_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(weekly_output, index=False)
    monthly.to_csv(monthly_output, index=False)

    print(f"Saved weekly features: {weekly_output} | rows={len(weekly):,}")
    print(f"Saved monthly features: {monthly_output} | rows={len(monthly):,}")


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    input_path = resolve_path(args.input, project_root)
    weekly_output = resolve_path(args.weekly_output, project_root)
    monthly_output = resolve_path(args.monthly_output, project_root)

    run(input_path, weekly_output, monthly_output)
