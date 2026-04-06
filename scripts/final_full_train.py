from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: Full-scale weekly/monthly two-stage training on engineered features"
    )
    parser.add_argument(
        "--weekly-input",
        type=Path,
        default=Path("data/processed/final_weekly_features.csv"),
        help="Weekly feature dataset path",
    )
    parser.add_argument(
        "--monthly-input",
        type=Path,
        default=Path("data/processed/final_monthly_features.csv"),
        help="Monthly feature dataset path",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/dissanayaka_master_model.pkl"),
        help="Path to save trained full pipeline",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("exports/plots/final_forecast_vs_actual_top_product_2025.png"),
        help="Path to save forecast-vs-actual chart",
    )
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


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


def build_models(random_state: int = 42) -> tuple[LGBMClassifier, XGBRegressor, LGBMRegressor, RandomForestRegressor]:
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    xgb = XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )

    lgbm = LGBMRegressor(
        objective="regression_l1",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )

    return clf, xgb, lgbm, rf


def train_two_stage(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    train_df = df[(df[date_col] >= "2019-01-01") & (df[date_col] < "2025-01-01")].copy()
    test_df = df[(df[date_col] >= "2025-01-01") & (df[date_col] < "2026-01-01")].copy()

    if train_df.empty:
        raise ValueError("Training split is empty for 2019-2024.")
    if test_df.empty:
        raise ValueError("Test split is empty for 2025.")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df["Quantity"].to_numpy(dtype=np.float32)
    y_test = test_df["Quantity"].to_numpy(dtype=np.float32)

    y_train_binary = (y_train > 0).astype(np.int8)

    clf, xgb, lgbm, rf = build_models(random_state=42)
    clf.fit(X_train, y_train_binary)

    positive_mask = y_train > 0
    X_train_pos = X_train.loc[positive_mask]
    y_train_pos = y_train[positive_mask]
    if X_train_pos.empty:
        raise ValueError("No positive-sales rows for stage-2 training.")

    xgb.fit(X_train_pos, y_train_pos)
    lgbm.fit(X_train_pos, y_train_pos)
    rf.fit(X_train_pos, y_train_pos)

    demand_pred = clf.predict(X_test)
    pred_qty = np.zeros(len(X_test), dtype=np.float64)

    pos_idx = np.where(demand_pred == 1)[0]
    if len(pos_idx) > 0:
        X_pos = X_test.iloc[pos_idx]
        p_xgb = xgb.predict(X_pos)
        p_lgbm = lgbm.predict(X_pos)
        p_rf = rf.predict(X_pos)
        blended = 0.45 * p_xgb + 0.45 * p_lgbm + 0.10 * p_rf
        pred_qty[pos_idx] = np.clip(blended, a_min=0.0, a_max=None)

    # Business rule: quantity predictions must be rounded to whole numbers.
    pred_qty = np.rint(pred_qty).astype(np.int32)

    pred_df = test_df[[date_col, "ProductID", "ProductName", "Category"]].copy()
    pred_df = pred_df.rename(columns={date_col: "Date"})
    pred_df["Actual_Quantity"] = y_test
    pred_df["Predicted_Quantity"] = pred_qty

    metrics = metric_pack(
        pred_df["Actual_Quantity"].to_numpy(dtype=np.float64),
        pred_df["Predicted_Quantity"].to_numpy(dtype=np.float64),
    )

    return {
        "classifier": clf,
        "xgb_regressor": xgb,
        "lgbm_regressor": lgbm,
        "rf_regressor": rf,
        "predictions": pred_df,
        "metrics": metrics,
        "feature_columns": feature_cols,
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


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    weekly_input = resolve_path(args.weekly_input, project_root)
    monthly_input = resolve_path(args.monthly_input, project_root)
    model_output = resolve_path(args.model_output, project_root)
    plot_output = resolve_path(args.plot_output, project_root)

    weekly_df = load_feature_data(weekly_input, date_col="Week_Ending_Sunday")
    monthly_df = load_feature_data(monthly_input, date_col="Month_Start")

    weekly_features = build_feature_columns(weekly_df)
    monthly_features = build_feature_columns(monthly_df)
    if not weekly_features:
        raise ValueError("No numeric weekly feature columns found.")
    if not monthly_features:
        raise ValueError("No numeric monthly feature columns found.")

    weekly_result = train_two_stage(weekly_df, date_col="Week_Ending_Sunday", feature_cols=weekly_features)
    monthly_result = train_two_stage(monthly_df, date_col="Month_Start", feature_cols=monthly_features)

    table = pd.DataFrame(
        [
            {"Horizon": "Weekly", **weekly_result["metrics"]},
            {"Horizon": "Monthly", **monthly_result["metrics"]},
        ]
    )["Horizon R2 MAE RMSE MAPE".split()]

    print("\nFull Scale Accuracy Table")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_top_product(weekly_result["predictions"], monthly_result["predictions"], plot_output)
    print(f"\nSaved forecast plot: {plot_output}")

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
    print(f"Saved model: {model_output}")


if __name__ == "__main__":
    main()
