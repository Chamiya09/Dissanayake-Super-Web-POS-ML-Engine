from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
    item_column: str
    feature_columns: list[str]
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    r2: float
    mape: float
    best_params: dict[str, object]


class DemandForecaster:
    """Weekly demand forecasting pipeline using top-seller filtering + XGBoost."""

    def __init__(
        self,
        input_path: Path,
        model_output_path: Path,
        top_n_items: int = 50,
        train_ratio: float = 0.8,
    ) -> None:
        self.input_path = input_path
        self.model_output_path = model_output_path
        self.top_n_items = top_n_items
        self.train_ratio = train_ratio

        self.date_col = "Date"
        self.target_col = "Quantity"
        self.product_col = "Item_ID"
        self.price_col = "SellingPrice"

        self.df_raw: pd.DataFrame | None = None
        self.df_weekly: pd.DataFrame | None = None
        self.df_features: pd.DataFrame | None = None
        self.model: XGBRegressor | None = None
        self.best_params: dict[str, object] = {}
        self.feature_columns: list[str] = []

    def run(self) -> TrainingArtifacts:
        """Execute full training pipeline and return artifacts."""
        self.load_data()
        self.filter_top_sellers()
        self.aggregate_weekly()
        self.create_weekly_features()
        X_train, X_test, y_train, y_test = self.time_series_split()
        self.train_model(X_train, y_train)
        mae, rmse, r2, mape = self.evaluate(X_test, y_test)
        self.save_model()

        return TrainingArtifacts(
            model=self.model,
            item_column=self.product_col,
            feature_columns=self.feature_columns,
            train_rows=len(X_train),
            test_rows=len(X_test),
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            best_params=self.best_params,
        )

    def load_data(self) -> None:
        """Read processed CSV and normalize required schema."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        self.df_raw = pd.read_csv(self.input_path)
        self.df_raw.columns = self.df_raw.columns.str.strip()

        required = {self.date_col, self.target_col}
        missing_required = required - set(self.df_raw.columns)
        if missing_required:
            raise KeyError(f"Missing required columns: {sorted(missing_required)}")

        if self.product_col not in self.df_raw.columns:
            product_fallbacks = ["ProductName", "Product_ID", "SKU", "ItemId"]
            fallback = next(
                (col for col in product_fallbacks if col in self.df_raw.columns), None
            )
            if fallback is None:
                raise KeyError(
                    "Could not find product identifier column. "
                    f"Checked: {[self.product_col, *product_fallbacks]}"
                )
            self.product_col = fallback

        if self.price_col not in self.df_raw.columns:
            price_fallbacks = ["Price", "UnitPrice", "Selling_Price"]
            price_fallback = next(
                (col for col in price_fallbacks if col in self.df_raw.columns), None
            )
            if price_fallback is None:
                raise KeyError(
                    "Could not find price column. "
                    f"Checked: {[self.price_col, *price_fallbacks]}"
                )
            self.price_col = price_fallback

        self.df_raw[self.date_col] = pd.to_datetime(
            self.df_raw[self.date_col], errors="coerce"
        )
        self.df_raw[self.target_col] = pd.to_numeric(
            self.df_raw[self.target_col], errors="coerce"
        )
        self.df_raw[self.price_col] = pd.to_numeric(
            self.df_raw[self.price_col], errors="coerce"
        )
        self.df_raw = self.df_raw.dropna(subset=[self.date_col, self.target_col]).copy()

    def filter_top_sellers(self) -> None:
        """Keep only top-N items by lifetime quantity to reduce intermittent-demand noise."""
        self._ensure(self.df_raw, "Raw data is not loaded. Call load_data() first.")

        item_totals = (
            self.df_raw.groupby(self.product_col, as_index=False)[self.target_col]
            .sum()
            .sort_values(self.target_col, ascending=False)
        )
        top_items = item_totals.head(self.top_n_items)[self.product_col]
        self.df_raw = self.df_raw[self.df_raw[self.product_col].isin(top_items)].copy()

        if self.df_raw.empty:
            raise ValueError("No rows left after top-seller filtering.")

    def aggregate_weekly(self) -> None:
        """Aggregate weekly item-level quantity sum and selling price mean using W-MON."""
        self._ensure(self.df_raw, "Raw data is not loaded. Call load_data() first.")

        weekly = (
            self.df_raw.groupby(
                [self.product_col, pd.Grouper(key=self.date_col, freq="W-MON")],
                as_index=False,
            )
            .agg(
                Quantity=(self.target_col, "sum"),
                SellingPrice=(self.price_col, "mean"),
            )
            .sort_values([self.date_col, self.product_col])
            .reset_index(drop=True)
        )

        weekly["SellingPrice"] = pd.to_numeric(weekly["SellingPrice"], errors="coerce")
        weekly["SellingPrice"] = weekly["SellingPrice"].fillna(0)

        if weekly.empty:
            raise ValueError("Weekly aggregation produced an empty dataset.")

        self.df_weekly = weekly

    def create_weekly_features(self) -> None:
        """Create weekly lag, rolling, and calendar features."""
        self._ensure(
            self.df_weekly,
            "Weekly data is not available. Call aggregate_weekly() first.",
        )

        df = self.df_weekly.copy()

        # Lags from prior weeks.
        df["lag_1_week"] = df.groupby(self.product_col)[self.target_col].shift(1)
        df["lag_2_weeks"] = df.groupby(self.product_col)[self.target_col].shift(2)

        # Rolling mean based only on historical values to avoid leakage.
        df["rolling_mean_4_weeks"] = (
            df.groupby(self.product_col)[self.target_col]
            .transform(lambda s: s.shift(1).rolling(window=4, min_periods=4).mean())
        )

        # Calendar features from week timestamp.
        df["Month"] = df[self.date_col].dt.month
        df["WeekOfYear"] = df[self.date_col].dt.isocalendar().week.astype(int)

        required_feature_cols = [
            "lag_1_week",
            "lag_2_weeks",
            "rolling_mean_4_weeks",
            "SellingPrice",
        ]
        df = df.dropna(subset=required_feature_cols).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No rows left after weekly feature engineering. "
                "Check data volume or reduce lag/rolling windows."
            )

        self.df_features = df
        self.feature_columns = [
            "lag_1_week",
            "lag_2_weeks",
            "rolling_mean_4_weeks",
            "SellingPrice",
            "Month",
            "WeekOfYear",
        ]

    def time_series_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split chronologically by week: first 80% train, last 20% test."""
        self._ensure(
            self.df_features,
            "Feature data is not available. Call create_weekly_features() first.",
        )

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        df = self.df_features.sort_values(self.date_col).copy()
        unique_weeks = np.array(sorted(df[self.date_col].unique()))

        if len(unique_weeks) < 2:
            raise ValueError("Need at least 2 unique weeks to create train/test split.")

        split_idx = int(len(unique_weeks) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(unique_weeks) - 1)
        split_week = unique_weeks[split_idx]

        train_df = df[df[self.date_col] < split_week].copy()
        test_df = df[df[self.date_col] >= split_week].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                "Chronological weekly split produced empty train or test set. "
                "Adjust train_ratio or verify date range."
            )

        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_col]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.target_col]

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Tune XGBoost with GridSearchCV + TimeSeriesSplit and keep best model."""
        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )

        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4, 5],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_params = dict(grid.best_params_)

        print(f"Best params from GridSearchCV: {self.best_params}")

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> tuple[float, float, float, float]:
        """Calculate MAE, RMSE, R-squared, and safe MAPE on test data."""
        self._ensure(self.model, "Model is not trained. Call train_model() first.")

        y_pred = self.model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        non_zero_mask = y_test.to_numpy() != 0
        if non_zero_mask.any():
            mape = float(
                mean_absolute_percentage_error(
                    y_test.to_numpy()[non_zero_mask],
                    y_pred[non_zero_mask],
                )
                * 100
            )
        else:
            mape = 0.0

        print("Evaluation on chronological weekly test set")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R-squared : {r2:.4f}")
        print(f"MAPE : {mape:.4f}%")

        return mae, rmse, r2, mape

    def save_model(self) -> None:
        """Persist best model artifact to the configured output path."""
        self._ensure(self.model, "Model is not trained. Cannot save model.")

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
    output_model_file = project_root / "models" / "xgboost_champion.pkl"

    forecaster = DemandForecaster(
        input_path=input_file,
        model_output_path=output_model_file,
        top_n_items=50,
        train_ratio=0.8,
    )

    artifacts = forecaster.run()

    print("Training pipeline completed successfully.")
    print(f"Item column used: {artifacts.item_column}")
    print(f"Train rows: {artifacts.train_rows}")
    print(f"Test rows : {artifacts.test_rows}")
    print(f"Features  : {artifacts.feature_columns}")
    print(f"Best params: {artifacts.best_params}")
    print(f"MAE       : {artifacts.mae:.4f}")
    print(f"RMSE      : {artifacts.rmse:.4f}")
    print(f"R-squared : {artifacts.r2:.4f}")
    print(f"MAPE      : {artifacts.mape:.4f}%")


if __name__ == "__main__":
    main()
