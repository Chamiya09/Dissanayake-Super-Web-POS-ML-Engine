from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
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
