from __future__ import annotations

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

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for training. Install it with: pip install xgboost"
    ) from exc


class DemandForecaster:
    """All-in-one demand forecasting trainer for daily, weekly, and monthly models."""

    def __init__(
        self,
        input_path: Path,
        top_n_items: int = 50,
        train_ratio: float = 0.8,
    ) -> None:
        self.input_path = input_path
        self.top_n_items = top_n_items
        self.train_ratio = train_ratio

        self.date_col = "Date"
        self.target_col = "Quantity"
        self.log_target_col = "log_Quantity"
        self.product_col = "Item_ID"
        self.price_col = "SellingPrice"

        self.df_raw: pd.DataFrame | None = None

    def run(self) -> None:
        """Load/filter source data once, then train all configured frequencies."""
        self.load_data()
        self.filter_top_sellers()

    def load_data(self) -> None:
        """Load CSV, normalize schema, and coerce data types."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        df.columns = df.columns.str.strip()

        required = {self.date_col, self.target_col}
        missing_required = required - set(df.columns)
        if missing_required:
            raise KeyError(f"Missing required columns: {sorted(missing_required)}")

        if self.product_col not in df.columns:
            product_fallbacks = ["ProductName", "Product_ID", "SKU", "ItemId"]
            fallback = next((col for col in product_fallbacks if col in df.columns), None)
            if fallback is None:
                raise KeyError(
                    "Could not find product identifier column. "
                    f"Checked: {[self.product_col, *product_fallbacks]}"
                )
            self.product_col = fallback

        if self.price_col not in df.columns:
            price_fallbacks = ["Price", "UnitPrice", "Selling_Price"]
            fallback = next((col for col in price_fallbacks if col in df.columns), None)
            if fallback is None:
                raise KeyError(
                    "Could not find price column. "
                    f"Checked: {[self.price_col, *price_fallbacks]}"
                )
            self.price_col = fallback

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")
        df[self.price_col] = pd.to_numeric(df[self.price_col], errors="coerce")
        df = df.dropna(subset=[self.date_col, self.target_col, self.product_col]).copy()

        if df.empty:
            raise ValueError("Input dataset is empty after type coercion and null filtering.")

        self.df_raw = df

    def filter_top_sellers(self) -> None:
        """Keep only top-N best-selling items by lifetime quantity from raw data."""
        self._ensure(self.df_raw, "Raw data is not loaded. Call load_data() first.")

        totals = (
            self.df_raw.groupby(self.product_col, as_index=False)[self.target_col]
            .sum()
            .sort_values(self.target_col, ascending=False)
        )
        top_items = totals.head(self.top_n_items)[self.product_col]

        self.df_raw = self.df_raw[self.df_raw[self.product_col].isin(top_items)].copy()
        if self.df_raw.empty:
            raise ValueError("No rows left after filtering top-selling items.")

    def _aggregate_by_frequency(self, frequency: str) -> pd.DataFrame:
        """Aggregate quantity and price by item and requested time frequency."""
        self._ensure(self.df_raw, "Raw data is not loaded. Call load_data() first.")

        aggregated = (
            self.df_raw.groupby(
                [self.product_col, pd.Grouper(key=self.date_col, freq=frequency)],
                as_index=False,
            )
            .agg(
                Quantity=(self.target_col, "sum"),
                SellingPrice=(self.price_col, "mean"),
            )
            .reset_index(drop=True)
        )

        aggregated[self.target_col] = pd.to_numeric(aggregated[self.target_col], errors="coerce")
        aggregated[self.price_col] = pd.to_numeric(aggregated[self.price_col], errors="coerce")
        aggregated = aggregated.dropna(subset=[self.date_col, self.target_col, self.price_col]).copy()
        aggregated[self.price_col] = aggregated[self.price_col].fillna(0.0)
        aggregated = aggregated.sort_values([self.product_col, self.date_col]).reset_index(drop=True)

        if aggregated.empty:
            raise ValueError(f"Aggregation with frequency '{frequency}' returned an empty dataframe.")

        return aggregated

    def _create_features(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Create lag/rolling and calendar features with strict item-level grouping."""
        df = df.sort_values([self.product_col, self.date_col]).reset_index(drop=True)

        item_group = df.groupby(self.product_col)
        df["lag_1"] = item_group[self.target_col].shift(1)
        df["lag_2"] = item_group[self.target_col].shift(2)
        df["lag_3"] = item_group[self.target_col].shift(3)
        df["lag_4"] = item_group[self.target_col].shift(4)
        df["rolling_mean_4_weeks"] = item_group[self.target_col].transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=4).mean()
        )

        df["Month"] = df[self.date_col].dt.month.astype(int)

        if frequency == "D":
            df["DayOfWeek"] = df[self.date_col].dt.dayofweek.astype(int)
        if frequency.startswith("W"):
            df["WeekOfYear"] = df[self.date_col].dt.isocalendar().week.astype(int)

        df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4_weeks"]).reset_index(
            drop=True
        )

        if df.empty:
            raise ValueError(
                "No rows left after creating lag/rolling features. "
                "Increase history or reduce lag/rolling windows."
            )

        if (df[self.target_col] < 0).any():
            raise ValueError("Quantity contains negative values; log1p transform requires non-negative target.")

        df[self.log_target_col] = np.log1p(df[self.target_col])
        return df

    def _one_hot_encode_items(self, df: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, list[str]]:
        """One-hot encode item identity and build feature column list per frequency."""
        item_dummies = pd.get_dummies(df[self.product_col], prefix="item", dtype=int)
        df = pd.concat([df, item_dummies], axis=1)

        feature_columns = [
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "rolling_mean_4_weeks",
            "SellingPrice",
            "Month",
        ]

        if frequency == "D":
            feature_columns.append("DayOfWeek")
        if frequency.startswith("W"):
            feature_columns.append("WeekOfYear")

        feature_columns.extend(item_dummies.columns.tolist())
        return df, feature_columns

    def _time_series_split(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Chronological split (80/20) using log-transformed target."""

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        df = df.sort_values(self.date_col).reset_index(drop=True)
        split_idx = int(len(df) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(df) - 1)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Chronological split produced an empty train or test set.")

        X_train = train_df[feature_columns]
        y_train = train_df[self.log_target_col]
        X_test = test_df[feature_columns]
        y_test = test_df[self.log_target_col]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _build_model() -> XGBRegressor:
        """Create XGBoost regressor with stable fast defaults."""
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        )

    @staticmethod
    def _evaluate(
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[float, float, float, float]:
        """Inverse-transform predictions/targets and evaluate on the original scale."""
        y_pred_log = model.predict(X_test)
        y_true_log = y_test.to_numpy()

        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_true_log)

        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        y_true = np.clip(y_true, a_min=0.0, a_max=None)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

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

        return mae, rmse, r2, mape

    def train_and_save(self, frequency: str, output_path: Path) -> None:
        """Train and save one model for a given frequency string."""
        if self.df_raw is None:
            raise ValueError("Raw data is not prepared. Call run() first.")

        aggregated = self._aggregate_by_frequency(frequency)
        features_df = self._create_features(aggregated, frequency)
        encoded_df, feature_columns = self._one_hot_encode_items(features_df, frequency)
        X_train, X_test, y_train, y_test = self._time_series_split(encoded_df, feature_columns)

        model = self._build_model()
        model.fit(X_train, y_train)

        mae, rmse, r2, mape = self._evaluate(model, X_test, y_test)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

        print(f"\n[{frequency}] Model trained and saved")
        print(f"Output file: {output_path}")
        print(f"Train rows : {len(X_train)}")
        print(f"Test rows  : {len(X_test)}")
        print(f"Features   : {len(feature_columns)}")
        print(f"MAE        : {mae:.4f}")
        print(f"RMSE       : {rmse:.4f}")
        print(f"R-squared  : {r2:.4f}")
        print(f"MAPE       : {mape:.4f}%")

    @staticmethod
    def _ensure(value: object, message: str) -> None:
        if value is None:
            raise ValueError(message)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "processed" / "cleaned_pos_data.csv"
    output_daily = project_root / "models" / "xgboost_daily.pkl"
    output_weekly = project_root / "models" / "xgboost_weekly.pkl"
    output_monthly = project_root / "models" / "xgboost_monthly.pkl"

    forecaster = DemandForecaster(input_path=input_file, top_n_items=50, train_ratio=0.8)
    forecaster.run()

    training_plan: list[tuple[str, Path]] = [
        ("D", output_daily),
        ("W-MON", output_weekly),
        ("MS", output_monthly),
    ]

    for frequency, output_path in training_plan:
        forecaster.train_and_save(frequency=frequency, output_path=output_path)

    print("\nAll frequency models trained and saved successfully.")


if __name__ == "__main__":
    main()
