from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def plot_monthly_revenue_trend(df: pd.DataFrame, plots_dir: Path) -> None:
    monthly_revenue = (
        df.assign(Month=df["Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)["Total_LKR"]
        .sum()
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly_revenue, x="Month", y="Total_LKR", linewidth=2.2, color="#2a9d8f")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Revenue (LKR)")
    plt.tight_layout()
    plt.savefig(plots_dir / "monthly_revenue_trend.png", dpi=150)
    plt.close()


def plot_category_revenue_distribution(df: pd.DataFrame, plots_dir: Path) -> None:
    category_revenue = (
        df.groupby("Category", as_index=False)["Total_LKR"]
        .sum()
        .sort_values("Total_LKR", ascending=False)
    )

    plt.figure(figsize=(14, 7))
    sns.barplot(data=category_revenue, x="Category", y="Total_LKR", color="#457b9d")
    plt.title("Category Revenue Distribution")
    plt.xlabel("Category")
    plt.ylabel("Total Revenue (LKR)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "category_revenue_distribution.png", dpi=150)
    plt.close()


def plot_hourly_transaction_volume(df: pd.DataFrame, plots_dir: Path) -> None:
    hourly = (
        df.assign(Hour=df["TransactionDateTime"].dt.hour)
        .groupby("Hour", as_index=False)["TransactionID"]
        .nunique()
        .rename(columns={"TransactionID": "Unique_Transactions"})
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly, x="Hour", y="Unique_Transactions", marker="o", color="#e76f51")
    plt.title("Hourly Transaction Volume")
    plt.xlabel("Hour of Day")
    plt.ylabel("Unique Transactions")
    plt.tight_layout()
    plt.savefig(plots_dir / "hourly_transaction_volume.png", dpi=150)
    plt.close()


def plot_weekday_hour_quantity_heatmap(df: pd.DataFrame, plots_dir: Path) -> None:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    heatmap_df = (
        df.assign(
            Weekday=df["TransactionDateTime"].dt.day_name(),
            Hour=df["TransactionDateTime"].dt.hour,
        )
        .groupby(["Weekday", "Hour"], as_index=False)["Quantity"]
        .sum()
    )

    pivot = heatmap_df.pivot(index="Weekday", columns="Hour", values="Quantity")
    pivot = pivot.reindex(weekday_order).fillna(0)

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.2)
    plt.title("Weekday-Hour Quantity Heatmap")
    plt.xlabel("Hour of Day")
    plt.ylabel("Weekday")
    plt.tight_layout()
    plt.savefig(plots_dir / "weekday_hour_quantity_heatmap.png", dpi=150)
    plt.close()


def plot_unit_margin_distribution(df: pd.DataFrame, plots_dir: Path) -> None:
    margin_df = df.assign(Unit_Margin=df["SellingPrice"] - df["BuyingPrice"])

    plt.figure(figsize=(12, 6))
    sns.histplot(margin_df["Unit_Margin"], bins=40, color="#6d597a", edgecolor="black")
    plt.title("Unit Margin Distribution")
    plt.xlabel("SellingPrice - BuyingPrice")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "unit_margin_distribution.png", dpi=150)
    plt.close()


def run_eda_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_inflation_check(df, plots_dir)
    plot_seasonality_check(df, plots_dir)
    plot_basket_analysis(df, plots_dir)
    plot_top_items(df, plots_dir)
    plot_monthly_revenue_trend(df, plots_dir)
    plot_category_revenue_distribution(df, plots_dir)
    plot_hourly_transaction_volume(df, plots_dir)
    plot_weekday_hour_quantity_heatmap(df, plots_dir)
    plot_unit_margin_distribution(df, plots_dir)
