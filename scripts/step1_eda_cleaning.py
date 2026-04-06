from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: Data cleaning and advanced EDA")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/DISSANAYAKA_POS_DATASET_2019_ONWARDS.csv"),
        help="Path to raw POS CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cleaned_pos_data.csv"),
        help="Path to cleaned CSV output",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("exports/plots"),
        help="Directory for exported PNG plots",
    )
    return parser.parse_args()


def resolve_input_path(input_path: Path) -> Path:
    if input_path.exists():
        return input_path

    # Fallback to first matching dataset if the exact filename differs.
    candidates = sorted(input_path.parent.glob("DISSANAYAKA_POS_DATASET_*.csv"))
    if candidates:
        print(
            f"[WARN] Input not found at {input_path}. "
            f"Using fallback dataset: {candidates[0]}"
        )
        return candidates[0]

    raise FileNotFoundError(f"Raw dataset not found: {input_path}")


def round_to_nearest_10(series: pd.Series) -> pd.Series:
    return (np.round(series.astype(float) / 10.0) * 10.0).astype(int)


def clean_pos_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required_cols = {
        "TransactionID",
        "Date",
        "Time",
        "ProductName",
        "Quantity",
        "SellingPrice",
        "BuyingPrice",
        "Total_LKR",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Normalize column names and strip string values.
    df.columns = [c.strip() for c in df.columns]
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    # Parse date and time robustly.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    time_parsed = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
    df["Time"] = time_parsed.dt.strftime("%H:%M:%S")
    df["DateTime"] = pd.to_datetime(
        df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"],
        errors="coerce",
    )

    # Numeric coercion.
    numeric_cols = ["Quantity", "SellingPrice", "BuyingPrice", "Total_LKR"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Null handling: drop rows missing critical business fields, impute others.
    critical_cols = ["TransactionID", "Date", "Time", "DateTime", "ProductName", "Quantity"]
    before_drop = len(df)
    df = df.dropna(subset=critical_cols).copy()
    dropped_rows = before_drop - len(df)

    # Impute numeric non-critical values using column medians.
    for col in ["SellingPrice", "BuyingPrice", "Total_LKR"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # If any category-like columns still have nulls, fill with Unknown.
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    # Ensure money columns have no decimals; if decimals exist, round to nearest 10 and cast to int.
    rounded_counts: dict[str, int] = {}
    for col in ["SellingPrice", "BuyingPrice", "Total_LKR"]:
        has_decimal = ~np.isclose(df[col].to_numpy() % 1, 0.0)
        decimal_count = int(has_decimal.sum())
        rounded_counts[col] = decimal_count
        if decimal_count > 0:
            df[col] = round_to_nearest_10(df[col])
        else:
            df[col] = df[col].round().astype(int)

    # Quantity is discrete by business meaning.
    df["Quantity"] = df["Quantity"].round().astype(int)

    # Chronological integrity check: transaction numeric id should not go backward over time.
    tx_audit = (
        df.groupby("TransactionID", as_index=False)["DateTime"]
        .min()
        .sort_values("DateTime")
        .reset_index(drop=True)
    )
    tx_audit["TxnNum"] = (
        tx_audit["TransactionID"].astype(str).str.extract(r"(\d+)", expand=False)
    )
    tx_audit["TxnNum"] = pd.to_numeric(tx_audit["TxnNum"], errors="coerce")
    tx_audit = tx_audit.dropna(subset=["TxnNum"]).copy()
    tx_audit["TxnNumDiff"] = tx_audit["TxnNum"].diff()
    chronology_violations = int((tx_audit["TxnNumDiff"] < 0).sum())

    audit_summary = {
        "rows_after_cleaning": len(df),
        "rows_dropped_for_critical_nulls": dropped_rows,
        "chronology_violations": chronology_violations,
        "sellingprice_decimal_rows_rounded": rounded_counts["SellingPrice"],
        "buyingprice_decimal_rows_rounded": rounded_counts["BuyingPrice"],
        "totallkr_decimal_rows_rounded": rounded_counts["Total_LKR"],
    }

    return df, audit_summary


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120


def plot_inflation_audit(df: pd.DataFrame, plots_dir: Path) -> None:
    monthly_price = (
        df.set_index("Date")
        .resample("MS")["SellingPrice"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly_price, x="Date", y="SellingPrice", linewidth=2.5, color="#1f77b4")
    plt.title("Inflation Audit: Mean Monthly SellingPrice (2019-Present)")
    plt.xlabel("Month")
    plt.ylabel("Mean SellingPrice (LKR)")
    plt.tight_layout()
    plt.savefig(plots_dir / "inflation_audit_mean_monthly_selling_price.png")
    plt.close()


def plot_seasonality_heatmap(df: pd.DataFrame, plots_dir: Path) -> None:
    seasonality = df.copy()
    seasonality["MonthName"] = seasonality["Date"].dt.month_name().str.slice(0, 3)
    seasonality["DayOfWeekName"] = seasonality["Date"].dt.day_name().str.slice(0, 3)

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pivot = (
        seasonality.pivot_table(
            index="MonthName",
            columns="DayOfWeekName",
            values="Quantity",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=month_order, columns=dow_order)
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5, annot=False)
    plt.title("Seasonality Heatmap: Total Quantity by Month vs Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.savefig(plots_dir / "seasonality_heatmap_month_vs_dayofweek_quantity.png")
    plt.close()


def plot_pareto_top_products(df: pd.DataFrame, plots_dir: Path) -> None:
    top_products = (
        df.groupby("ProductName", as_index=False)["Total_LKR"]
        .sum()
        .sort_values("Total_LKR", ascending=False)
        .head(30)
    )

    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=top_products,
        x="Total_LKR",
        y="ProductName",
        orient="h",
        color="#4c72b0",
    )
    plt.title("Pareto Distribution: Top 30 Products by Total Revenue")
    plt.xlabel("Total Revenue (LKR)")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(plots_dir / "pareto_top30_products_revenue.png")
    plt.close()


def plot_basket_size_histogram(df: pd.DataFrame, plots_dir: Path) -> None:
    basket_sizes = df.groupby("TransactionID").size().rename("ItemsPerTransaction")

    plt.figure(figsize=(12, 6))
    sns.histplot(basket_sizes, bins=range(1, 12), kde=False, color="#2ca02c")
    plt.title("Basket Size Analysis: Items per Transaction")
    plt.xlabel("Number of Items per Transaction")
    plt.ylabel("Transaction Count")
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(plots_dir / "basket_size_histogram_items_per_transaction.png")
    plt.close()


def plot_daily_sales_volume(df: pd.DataFrame, plots_dir: Path) -> None:
    daily_qty = (
        df.groupby("Date", as_index=False)["Quantity"]
        .sum()
        .sort_values("Date")
    )

    plt.figure(figsize=(16, 6))
    sns.lineplot(data=daily_qty, x="Date", y="Quantity", linewidth=1.8, color="#d62728")
    plt.title("Daily Sales Volume: Total Quantity Sold")
    plt.xlabel("Date")
    plt.ylabel("Total Quantity")
    plt.tight_layout()
    plt.savefig(plots_dir / "daily_sales_volume_total_quantity.png")
    plt.close()


def generate_all_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    setup_plot_style()
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_inflation_audit(df, plots_dir)
    plot_seasonality_heatmap(df, plots_dir)
    plot_pareto_top_products(df, plots_dir)
    plot_basket_size_histogram(df, plots_dir)
    plot_daily_sales_volume(df, plots_dir)


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args.input)
    output_path = args.output
    plots_dir = args.plots_dir

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading raw dataset: {input_path}")
    raw_df = pd.read_csv(input_path)

    cleaned_df, audit = clean_pos_data(raw_df)

    print("[INFO] Data quality audit summary:")
    for key, value in audit.items():
        print(f"  - {key}: {value}")

    generate_all_plots(cleaned_df, plots_dir)

    cleaned_df.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {output_path}")
    print(f"[INFO] Plot exports saved to: {plots_dir}")


if __name__ == "__main__":
    main()
