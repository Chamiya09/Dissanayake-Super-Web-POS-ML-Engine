from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for training. Install it with: pip install xgboost"
    ) from exc


@dataclass
class TrainingArtifacts:
    """Container for key outputs from model training."""

    model: XGBRegressor
    feature_columns: list[str]
    train_rows: int
    test_rows: int
    mae: float
    rmse: float


class DemandForecaster:
    """End-to-end time-series demand forecasting pipeline using XGBoost."""

    def __init__(
        self,
        input_path: Path,
        model_output_path: Path,
        lag_steps: Sequence[int] = (7, 14),
        train_ratio: float = 0.8,
    ) -> None:
        self.input_path = input_path
        self.model_output_path = model_output_path
        self.lag_steps = sorted(set(lag_steps))
        self.train_ratio = train_ratio

        self.date_col = "Date"
        self.product_col = "Item_ID"
        self.target_col = "Quantity"

        self.df_raw: pd.DataFrame | None = None
        self.df_aggregated: pd.DataFrame | None = None
        self.df_features: pd.DataFrame | None = None
        self.model: XGBRegressor | None = None
        self.feature_columns: list[str] = []

    def run(self) -> TrainingArtifacts:
        """Execute full pipeline from load to model serialization."""
        self.load_data()
        self.aggregate_daily_item_demand()
        self.create_lag_features()
        X_train, X_test, y_train, y_test = self.time_series_split()
        self.train_model(X_train, y_train)
        mae, rmse = self.evaluate(X_test, y_test)
        self.serialize_model()

        return TrainingArtifacts(
            model=self.model,
            feature_columns=self.feature_columns,
            train_rows=len(X_train),
            test_rows=len(X_test),
            mae=mae,
            rmse=rmse,
        )

    def load_data(self) -> None:
        """Load cleaned POS data and validate required schema."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        self.df_raw = pd.read_csv(self.input_path)

        required = {self.date_col, self.target_col}
        missing_required = required - set(self.df_raw.columns)
        if missing_required:
            raise KeyError(f"Missing required columns: {sorted(missing_required)}")

        if self.product_col not in self.df_raw.columns:
            product_fallbacks = ["Product_ID", "ProductName", "SKU", "ItemId"]
            fallback = next((c for c in product_fallbacks if c in self.df_raw.columns), None)
            if fallback is None:
                raise KeyError(
                    "Could not find item/product identifier column. "
                    f"Checked: {[self.product_col, *product_fallbacks]}"
                )
            self.product_col = fallback

        self.df_raw[self.date_col] = pd.to_datetime(
            self.df_raw[self.date_col], errors="coerce"
        )
        self.df_raw = self.df_raw.dropna(subset=[self.date_col]).copy()

    def aggregate_daily_item_demand(self) -> None:
        """Aggregate transaction-level data to daily item-level demand."""
        self._ensure(self.df_raw, "Raw data is not loaded. Call load_data() first.")

        grouped = (
            self.df_raw.groupby([self.date_col, self.product_col], as_index=False)[
                self.target_col
            ]
            .sum()
            .sort_values([self.date_col, self.product_col])
            .reset_index(drop=True)
        )

        self.df_aggregated = grouped

    def create_lag_features(self) -> None:
        """Generate lag-based features for each item and drop resulting NaNs."""
        self._ensure(
            self.df_aggregated,
            "Aggregated data is not available. Call aggregate_daily_item_demand() first.",
        )

        df = self.df_aggregated.copy()

        for lag in self.lag_steps:
            df[f"lag_{lag}"] = df.groupby(self.product_col)[self.target_col].shift(lag)

        # Time features are useful for capturing calendar effects.
        df["DayOfWeek"] = df[self.date_col].dt.dayofweek
        df["Is_Weekend"] = (df[self.date_col].dt.dayofweek >= 5).astype(int)

        lag_cols = [f"lag_{lag}" for lag in self.lag_steps]
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No rows left after lag feature creation. "
                "Try reducing lag steps or verify data volume."
            )

        self.df_features = df

    def time_series_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Chronologically split data using unique dates (no random sampling)."""
        self._ensure(
            self.df_features,
            "Feature data is not available. Call create_lag_features() first.",
        )

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        df = self.df_features.sort_values(self.date_col).copy()
        unique_dates = np.array(sorted(df[self.date_col].unique()))

        if len(unique_dates) < 2:
            raise ValueError("Need at least 2 unique dates to create train/test split.")

        split_idx = int(len(unique_dates) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
        split_date = unique_dates[split_idx]

        train_df = df[df[self.date_col] < split_date].copy()
        test_df = df[df[self.date_col] >= split_date].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                "Chronological split produced empty train or test set. "
                "Adjust train_ratio or verify date coverage."
            )

        lag_cols = [f"lag_{lag}" for lag in self.lag_steps]
        calendar_cols = ["DayOfWeek", "Is_Weekend"]
        self.feature_columns = lag_cols + calendar_cols

        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_col]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.target_col]

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost regressor on the training time window."""
        self.model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
        """Evaluate forecasting quality on holdout period."""
        self._ensure(self.model, "Model is not trained. Call train_model() first.")

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print("Evaluation on chronological test set")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")

        return float(mae), float(rmse)

    def serialize_model(self) -> None:
        """Persist trained model artifact for downstream inference."""
        self._ensure(self.model, "Model is not trained. Cannot serialize.")

        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_output_path)
        print(f"Model saved to: {self.model_output_path}")

    @staticmethod
    def _ensure(value: object, message: str) -> None:
        if value is None:
            raise ValueError(message)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "processed" / "cleaned_pos_data.csv"
    model_file = project_root / "models" / "xgboost_champion.pkl"

    forecaster = DemandForecaster(
        input_path=input_file,
        model_output_path=model_file,
        lag_steps=(7, 14),
        train_ratio=0.8,
    )
    artifacts = forecaster.run()

    print("Training pipeline completed successfully.")
    print(f"Train rows: {artifacts.train_rows}")
    print(f"Test rows : {artifacts.test_rows}")
    print(f"Features  : {artifacts.feature_columns}")


if __name__ == "__main__":
    main()