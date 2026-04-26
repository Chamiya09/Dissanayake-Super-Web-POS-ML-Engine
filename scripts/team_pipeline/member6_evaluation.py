from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0
    return float(mean_absolute_percentage_error(y_true[non_zero], y_pred[non_zero]) * 100.0)


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": safe_mape(y_true, y_pred),
    }


def plot_top_product(weekly_pred: pd.DataFrame, monthly_pred: pd.DataFrame, out_path: Path) -> None:
    combined = pd.concat([weekly_pred, monthly_pred], ignore_index=True)
    if combined.empty:
        print("[WARN] No prediction rows available for plotting.")
        return

    top_product = (
        combined.groupby(["ProductID", "ProductName"], as_index=False)["Actual_Quantity"]
        .sum()
        .sort_values("Actual_Quantity", ascending=False)
        .iloc[0]
    )
    pid = top_product["ProductID"]
    pname = top_product["ProductName"]

    w = weekly_pred[weekly_pred["ProductID"] == pid].sort_values("Date")
    m = monthly_pred[monthly_pred["ProductID"] == pid].sort_values("Date")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(w["Date"], w["Actual_Quantity"], label="Actual", linewidth=2)
    axes[0].plot(w["Date"], w["Predicted_Quantity"], label="Predicted", linewidth=2)
    axes[0].set_title(f"Weekly Forecast vs Actual - {pid} | {pname}")
    axes[0].set_ylabel("Quantity")
    axes[0].legend()

    axes[1].plot(m["Date"], m["Actual_Quantity"], label="Actual", linewidth=2)
    axes[1].plot(m["Date"], m["Predicted_Quantity"], label="Predicted", linewidth=2)
    axes[1].set_title(f"Monthly Forecast vs Actual - {pid} | {pname}")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Quantity")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_accuracy_table(weekly_metrics: dict[str, float], monthly_metrics: dict[str, float]) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {"Horizon": "Weekly", **weekly_metrics},
            {"Horizon": "Monthly", **monthly_metrics},
        ]
    )["Horizon R2 MAE RMSE MAPE".split()]
    return table


def print_model_performance(
    weekly_metrics: dict[str, float],
    monthly_metrics: dict[str, float],
) -> None:
    print("\n--- MODEL PERFORMANCE ---")
    print("Weekly Forecast")
    print(f"MAE: {weekly_metrics['MAE']:.4f}")
    print(f"RMSE: {weekly_metrics['RMSE']:.4f}")
    print(f"R-squared Score: {weekly_metrics['R2']:.4f}")
    print("\nMonthly Forecast")
    print(f"MAE: {monthly_metrics['MAE']:.4f}")
    print(f"RMSE: {monthly_metrics['RMSE']:.4f}")
    print(f"R-squared Score: {monthly_metrics['R2']:.4f}")
    print("--- END MODEL PERFORMANCE ---")


def plot_actual_vs_predicted_chart(
    weekly_pred: pd.DataFrame,
    monthly_pred: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame(columns=["Date", "Actual_Quantity", "Predicted_Quantity"])
        out = predictions.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])
        return (
            out.groupby("Date", as_index=False)[["Actual_Quantity", "Predicted_Quantity"]]
            .sum()
            .sort_values("Date")
        )

    weekly = aggregate(weekly_pred)
    monthly = aggregate(monthly_pred)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    chart_specs = [
        (axes[0], weekly, "Weekly Actual Sales vs Predicted Sales"),
        (axes[1], monthly, "Monthly Actual Sales vs Predicted Sales"),
    ]

    for ax, data, title in chart_specs:
        if data.empty:
            ax.text(0.5, 0.5, "No test predictions available", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.plot(
            data["Date"],
            data["Actual_Quantity"],
            label="Actual Sales",
            color="#0f766e",
            linewidth=2.4,
        )
        ax.plot(
            data["Date"],
            data["Predicted_Quantity"],
            label="Predicted Sales",
            color="#f97316",
            linewidth=2.2,
            linestyle="--",
        )
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Sales Quantity", fontsize=11)
        ax.legend(loc="best", frameon=True)
        ax.grid(True, alpha=0.28)

    fig.suptitle("Demand Forecasting Model Evaluation", fontsize=18, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_artifact(
    weekly_result: dict[str, Any],
    monthly_result: dict[str, Any],
    model_output: Path,
) -> None:
    model_output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "weekly": {
            "classifier": weekly_result["classifier"],
            "xgb_regressor": weekly_result["xgb_regressor"],
            "lgbm_regressor": weekly_result["lgbm_regressor"],
            "rf_regressor": weekly_result["rf_regressor"],
            "feature_columns": weekly_result["feature_columns"],
            "date_column": "Week_Ending_Sunday",
        },
        "monthly": {
            "classifier": monthly_result["classifier"],
            "xgb_regressor": monthly_result["xgb_regressor"],
            "lgbm_regressor": monthly_result["lgbm_regressor"],
            "rf_regressor": monthly_result["rf_regressor"],
            "feature_columns": monthly_result["feature_columns"],
            "date_column": "Month_Start",
        },
        "weights": {"xgb": 0.45, "lgbm": 0.45, "rf": 0.10},
        "training_window": "2019-01-01 to 2024-12-31",
        "test_window": "2025-01-01 to 2025-12-31",
        "prediction_rounding": "Nearest whole number",
    }
    joblib.dump(artifact, model_output)
