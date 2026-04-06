from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RAW_FALLBACK_NAME = "DISSANAYAKA_POS_DATASET_2019-2025.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Professional Data Cleaning and EDA"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/DISSANAYAKA_FINAL_DATASET.csv"),
        help="Raw input dataset path",
    )
    parser.add_argument(
        "--output-cleaned",
        type=Path,
        default=Path("data/processed/cleaned_pos_data.csv"),
        help="Output path for cleaned dataset",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("exports/plots"),
        help="Directory to save EDA plots",
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def resolve_input_with_fallback(input_path: Path) -> Path:
    if input_path.exists():
        return input_path

    fallback = input_path.parent / RAW_FALLBACK_NAME
    if fallback.exists():
        print(f"[WARN] Input not found at {input_path}. Using fallback: {fallback}")
        return fallback

    raise FileNotFoundError(f"Dataset not found: {input_path}")


def round_to_nearest_10(series: pd.Series) -> pd.Series:
    return (series.div(10).round().mul(10)).astype("int64")


def load_and_validate(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

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
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time

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


def plot_inflation_check(df: pd.DataFrame, plots_dir: Path) -> None:
    monthly = (
        df.assign(Month=df["Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)["SellingPrice"]
        .mean()
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly, x="Month", y="SellingPrice", linewidth=2.2, color="#d1495b")
    plt.title("Inflation Check: Average Monthly SellingPrice")
    plt.xlabel("Month")
    plt.ylabel("Avg SellingPrice")
    plt.tight_layout()
    plt.savefig(plots_dir / "inflation_check_monthly_selling_price.png", dpi=150)
    plt.close()


def plot_seasonality_check(df: pd.DataFrame, plots_dir: Path) -> None:
    daily = df.groupby("Date", as_index=False)["Quantity"].sum()

    plt.figure(figsize=(16, 6))
    sns.lineplot(data=daily, x="Date", y="Quantity", linewidth=1.8, color="#00798c")
    plt.title("Seasonality Check: Total Daily Quantity Sold")
    plt.xlabel("Date")
    plt.ylabel("Total Quantity Sold")
    plt.tight_layout()
    plt.savefig(plots_dir / "seasonality_daily_quantity.png", dpi=150)
    plt.close()


def plot_basket_analysis(df: pd.DataFrame, plots_dir: Path) -> None:
    basket_sizes = df.groupby("TransactionID").size().rename("items_per_transaction").reset_index()

    plt.figure(figsize=(10, 6))
    sns.histplot(
        basket_sizes["items_per_transaction"],
        bins=range(1, int(basket_sizes["items_per_transaction"].max()) + 2),
        color="#edae49",
        edgecolor="black",
    )
    plt.title("Basket Analysis: Distribution of Items per TransactionID")
    plt.xlabel("Items per Transaction")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "basket_analysis_items_per_transaction.png", dpi=150)
    plt.close()


def plot_top_items(df: pd.DataFrame, plots_dir: Path) -> None:
    top_items = (
        df.groupby("ProductName", as_index=False)["Total_LKR"]
        .sum()
        .sort_values("Total_LKR", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(14, 9))
    sns.barplot(data=top_items, x="Total_LKR", y="ProductName", color="#30638e")
    plt.title("Top 20 Items by Total Sales")
    plt.xlabel("Total Sales (LKR)")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(plots_dir / "top_20_items_total_sales.png", dpi=150)
    plt.close()


def run(input_path: Path, output_cleaned: Path, plots_dir: Path) -> None:
    cleaned = load_and_validate(input_path)
    cleaned = apply_price_cleaning(cleaned)

    plots_dir.mkdir(parents=True, exist_ok=True)
    output_cleaned.parent.mkdir(parents=True, exist_ok=True)

    plot_inflation_check(cleaned, plots_dir)
    plot_seasonality_check(cleaned, plots_dir)
    plot_basket_analysis(cleaned, plots_dir)
    plot_top_items(cleaned, plots_dir)

    cleaned.to_csv(output_cleaned, index=False)

    print(f"Saved cleaned dataset: {output_cleaned}")
    print(f"Saved plots directory: {plots_dir}")


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    input_path = resolve_input_with_fallback(resolve_path(args.input, project_root))
    output_cleaned = resolve_path(args.output_cleaned, project_root)
    plots_dir = resolve_path(args.plots_dir, project_root)

    run(input_path, output_cleaned, plots_dir)
