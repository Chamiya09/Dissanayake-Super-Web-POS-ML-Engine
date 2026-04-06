from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "ProductID",
    "ProductName",
    "Category",
    "Date",
    "Quantity",
    "SellingPrice",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2 - Advanced Multi-Level Feature Engineering"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cleaned_pos_data.csv"),
        help="Path to cleaned POS dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/final_features.csv"),
        help="Path to final engineered feature dataset",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Top N fast-moving items to keep",
    )
    return parser.parse_args()


def load_clean_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["SellingPrice"] = pd.to_numeric(df["SellingPrice"], errors="coerce")

    df = df.dropna(subset=["ProductID", "ProductName", "Category", "Date", "Quantity", "SellingPrice"])
    if df.empty:
        raise ValueError("Dataset is empty after coercion/null cleanup.")

    return df


def build_daily_panel(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Top fast-moving items by total lifetime demand.
    top_products = (
        df.groupby("ProductID", as_index=False)["Quantity"]
        .sum()
        .sort_values("Quantity", ascending=False)
        .head(top_n)["ProductID"]
    )

    top_df = df[df["ProductID"].isin(top_products)].copy()
    if top_df.empty:
        raise ValueError("No data found for selected top products.")

    daily = (
        top_df.groupby(["ProductID", "ProductName", "Category", "Date"], as_index=False)
        .agg(Quantity=("Quantity", "sum"), SellingPrice=("SellingPrice", "mean"))
        .sort_values(["ProductID", "Date"])
        .reset_index(drop=True)
    )

    # Stable metadata for each ProductID used during date expansion.
    product_meta = (
        top_df.groupby("ProductID", as_index=False)
        .agg(
            ProductName=("ProductName", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            Category=("Category", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        )
        .set_index("ProductID")
    )

    min_date = daily["Date"].min()
    max_date = daily["Date"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")

    full_index = pd.MultiIndex.from_product(
        [top_products.sort_values().tolist(), full_dates],
        names=["ProductID", "Date"],
    )

    panel = daily.set_index(["ProductID", "Date"]).reindex(full_index).reset_index()
    panel = panel.merge(product_meta, left_on="ProductID", right_index=True, how="left", suffixes=("", "_meta"))

    panel["ProductName"] = panel["ProductName"].fillna(panel["ProductName_meta"])
    panel["Category"] = panel["Category"].fillna(panel["Category_meta"])
    panel = panel.drop(columns=["ProductName_meta", "Category_meta"])

    # Continuity rule: unsold day => Quantity 0, price carried forward from previous day.
    panel["Quantity"] = panel["Quantity"].fillna(0.0)
    panel["SellingPrice"] = panel.groupby("ProductID")["SellingPrice"].transform(lambda s: s.ffill().bfill())

    # Final fallback only for products with all-null prices.
    panel["SellingPrice"] = panel["SellingPrice"].fillna(top_df["SellingPrice"].median())

    return panel.sort_values(["ProductID", "Date"]).reset_index(drop=True)


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("ProductID")

    # Daily/weekly/monthly lags.
    out["lag_1"] = g["Quantity"].shift(1)
    out["lag_2"] = g["Quantity"].shift(2)
    out["lag_3"] = g["Quantity"].shift(3)
    out["lag_7"] = g["Quantity"].shift(7)
    out["lag_14"] = g["Quantity"].shift(14)
    out["lag_30"] = g["Quantity"].shift(30)

    # Weekly/monthly trend features and weekly volatility.
    out["rolling_mean_7"] = g["Quantity"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=7).mean()
    )
    out["rolling_mean_30"] = g["Quantity"].transform(
        lambda s: s.shift(1).rolling(window=30, min_periods=30).mean()
    )
    out["rolling_std_7"] = g["Quantity"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=7).std()
    )

    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["DayOfWeek"] = out["Date"].dt.dayofweek.astype(int)
    out["Month"] = out["Date"].dt.month.astype(int)
    out["DayOfMonth"] = out["Date"].dt.day.astype(int)
    out["Is_Weekend"] = (out["DayOfWeek"] >= 5).astype(int)

    out["Is_Avurudu_Season"] = (
        (out["Date"].dt.month == 4)
        & (out["Date"].dt.day >= 1)
        & (out["Date"].dt.day <= 20)
    ).astype(int)
    out["Is_Vesak_Season"] = (out["Date"].dt.month == 5).astype(int)
    out["Is_Christmas_Season"] = (
        (out["Date"].dt.month == 12) & (out["Date"].dt.day >= 15)
    ).astype(int)

    return out


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if (out["Quantity"] < 0).any():
        raise ValueError("Quantity contains negative values; cannot apply log1p safely.")

    out["log_quantity"] = np.log1p(out["Quantity"])

    out = pd.get_dummies(out, columns=["Category"], prefix="Category", dtype=int)

    lag_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
        "rolling_std_7",
    ]
    out = out.dropna(subset=lag_cols).reset_index(drop=True)

    ordered_prefix = [
        "ProductID",
        "ProductName",
        "Date",
        "Quantity",
        "log_quantity",
        "SellingPrice",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
        "rolling_std_7",
        "DayOfWeek",
        "Month",
        "DayOfMonth",
        "Is_Weekend",
        "Is_Avurudu_Season",
        "Is_Vesak_Season",
        "Is_Christmas_Season",
    ]

    category_cols = sorted([c for c in out.columns if c.startswith("Category_")])
    remaining_cols = [c for c in out.columns if c not in ordered_prefix and c not in category_cols]

    final_cols = [c for c in ordered_prefix if c in out.columns] + category_cols + remaining_cols
    return out[final_cols]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading cleaned data from: {args.input}")
    clean_df = load_clean_data(args.input)

    print(f"[INFO] Building top-{args.top_n} daily panel with continuity")
    daily_panel = build_daily_panel(clean_df, top_n=args.top_n)

    print("[INFO] Creating hierarchical time-series features")
    featured = add_time_series_features(daily_panel)

    print("[INFO] Adding Sri Lankan temporal and seasonal features")
    featured = add_calendar_features(featured)

    print("[INFO] Applying log transform, category encoding, and lag NaN cleanup")
    final_df = finalize_dataset(featured)

    final_df.to_csv(args.output, index=False)
    print(f"[INFO] Final features saved to: {args.output}")
    print(f"[INFO] Final shape: {final_df.shape}")


if __name__ == "__main__":
    main()
