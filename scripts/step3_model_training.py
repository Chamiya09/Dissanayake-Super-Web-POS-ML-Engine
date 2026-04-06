from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for Step 3 training. Install it with: pip install xgboost"
    ) from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "lightgbm is required for Step 3 training. Install it with: pip install lightgbm"
    ) from exc


@dataclass
class MetricRow:
    level: str
    r2: float
    mae: float
    mape: float


class EnsembleDemandTrainer:
    """Train and evaluate a bottom-up daily ensemble for daily/weekly/monthly forecasting."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        train_ratio: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.train_ratio = train_ratio
        self.random_state = random_state

        self.date_col = "Date"
        self.id_col = "ProductID"
        self.target_col = "log_quantity"
        self.raw_target_col = "Quantity"

    def run(self) -> None:
        df = self.load_data()
        X_train, X_test, y_train, y_test, test_meta = self.prepare_data(df)

        model = self.build_ensemble()
        model.fit(X_train, y_train)

        daily_results = self.predict_daily(model, X_test, y_test, test_meta)
        metrics = self.evaluate_all_levels(daily_results)

        self.print_comparison_table(metrics)
        self.save_model(model, X_train.columns.tolist())

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input feature dataset not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        required_cols = {self.date_col, self.id_col, self.target_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")

        if self.raw_target_col in df.columns:
            df[self.raw_target_col] = pd.to_numeric(df[self.raw_target_col], errors="coerce")

        df = df.dropna(subset=[self.date_col, self.id_col, self.target_col]).copy()
        if df.empty:
            raise ValueError("Feature dataset is empty after basic cleanup.")

        return df

    def prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        ordered = df.sort_values(self.date_col).reset_index(drop=True)
        split_idx = int(len(ordered) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(ordered) - 1)

        train_df = ordered.iloc[:split_idx].copy()
        test_df = ordered.iloc[split_idx:].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Chronological split produced empty train or test sets.")

        exclude_cols = {
            self.date_col,
            self.id_col,
            self.target_col,
            self.raw_target_col,
            "ProductName",
        }

        candidate_cols = [c for c in train_df.columns if c not in exclude_cols]
        numeric_cols = train_df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            raise ValueError("No numeric feature columns found for training.")

        X_train = train_df[numeric_cols]
        y_train = train_df[self.target_col]
        X_test = test_df[numeric_cols]
        y_test = test_df[self.target_col]

        test_meta = test_df[[self.date_col, self.id_col]].copy()
        return X_train, X_test, y_train, y_test, test_meta

    def build_ensemble(self) -> VotingRegressor:
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=1,
        )

        lgbm_model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.random_state,
            n_jobs=-1,
        )

        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

        return VotingRegressor(
            estimators=[
                ("xgb", xgb_model),
                ("lgbm", lgbm_model),
                ("rf", rf_model),
            ]
        )

    def predict_daily(
        self,
        model: VotingRegressor,
        X_test: pd.DataFrame,
        y_test_log: pd.Series,
        test_meta: pd.DataFrame,
    ) -> pd.DataFrame:
        y_pred_log = model.predict(X_test)
        y_true_log = y_test_log.to_numpy()

        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_true_log)

        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        y_true = np.clip(y_true, a_min=0.0, a_max=None)

        results = test_meta.copy()
        results["Actual_Qty"] = y_true
        results["Predicted_Qty"] = y_pred
        return results.sort_values(self.date_col).reset_index(drop=True)

    def evaluate_all_levels(self, daily_results: pd.DataFrame) -> list[MetricRow]:
        daily_metrics = self.compute_metrics(
            daily_results["Actual_Qty"].to_numpy(),
            daily_results["Predicted_Qty"].to_numpy(),
            level="Daily",
        )

        weekly_rolled = self.rollup(daily_results, "W-MON")
        weekly_metrics = self.compute_metrics(
            weekly_rolled["Actual_Qty"].to_numpy(),
            weekly_rolled["Predicted_Qty"].to_numpy(),
            level="Weekly",
        )

        monthly_rolled = self.rollup(daily_results, "MS")
        monthly_metrics = self.compute_metrics(
            monthly_rolled["Actual_Qty"].to_numpy(),
            monthly_rolled["Predicted_Qty"].to_numpy(),
            level="Monthly",
        )

        return [daily_metrics, weekly_metrics, monthly_metrics]

    def rollup(self, daily_results: pd.DataFrame, freq: str) -> pd.DataFrame:
        temp = daily_results.copy()
        temp[self.date_col] = pd.to_datetime(temp[self.date_col], errors="coerce")

        rolled = (
            temp.groupby(self.id_col)
            .resample(freq, on=self.date_col)[["Actual_Qty", "Predicted_Qty"]]
            .sum()
            .reset_index()
        )
        return rolled

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, level: str) -> MetricRow:
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))

        non_zero_mask = y_true != 0
        if non_zero_mask.any():
            mape = float(
                mean_absolute_percentage_error(
                    y_true[non_zero_mask],
                    y_pred[non_zero_mask],
                )
                * 100
            )
        else:
            mape = 0.0

        return MetricRow(level=level, r2=r2, mae=mae, mape=mape)

    @staticmethod
    def print_comparison_table(rows: list[MetricRow]) -> None:
        table = pd.DataFrame(
            {
                "Level": [r.level for r in rows],
                "R2": [r.r2 for r in rows],
                "MAE": [r.mae for r in rows],
                "MAPE(%)": [r.mape for r in rows],
            }
        )

        print("\nAccuracy Comparison Table")
        print(table.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))

    def save_model(self, model: VotingRegressor, feature_columns: list[str]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact: dict[str, Any] = {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": self.target_col,
            "inverse_transform": "np.expm1",
            "model_type": "VotingRegressor(XGBoost, LightGBM, RandomForest)",
        }
        joblib.dump(artifact, self.output_path)
        print(f"\nModel artifact saved to: {self.output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "processed" / "final_features.csv"
    output_path = project_root / "models" / "demand_forecast_ensemble.pkl"

    trainer = EnsembleDemandTrainer(
        input_path=input_path,
        output_path=output_path,
        train_ratio=0.8,
        random_state=42,
    )
    trainer.run()


if __name__ == "__main__":
    main()
