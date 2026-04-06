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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for training. Install it with: pip install xgboost"
    ) from exc


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    mape: float


class AdvancedHierarchicalForecaster:
    """Bottom-up hierarchical forecasting using a daily base model and rolled-up evaluation."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        top_n_items: int = 50,
        train_ratio: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.top_n_items = top_n_items
        self.train_ratio = train_ratio
        self.random_state = random_state

        self.date_col = "Date"
        self.target_col = "Quantity"
        self.log_target_col = "log_Quantity"
        self.product_col = "Item_ID"
        self.price_col = "SellingPrice"

        self.df_raw: pd.DataFrame | None = None
        self.df_daily: pd.DataFrame | None = None
        self.df_features: pd.DataFrame | None = None
        self.feature_columns: list[str] = []

        self.model: XGBRegressor | None = None
        self.best_params: dict[str, object] = {}

    def run(self) -> None:
        """Execute the full extreme forecasting pipeline."""
        self.load_data()
        self.filter_top_sellers()
        self.aggregate_daily_with_calendar_fill()
        self.create_daily_features()
        self.apply_log_transform_and_encoding()

        X_train, X_test, y_train, y_test, test_index_frame = self.chronological_split()
        self.train_with_randomized_search(X_train, y_train)

        daily_eval_frame = self.predict_and_prepare_daily_eval_frame(
            X_test=X_test,
            y_test_log=y_test,
            test_index_frame=test_index_frame,
        )

        daily_metrics = self.compute_metrics(
            daily_eval_frame["actual"].to_numpy(),
            daily_eval_frame["predicted"].to_numpy(),
        )
        weekly_metrics = self.compute_rollup_metrics(daily_eval_frame, "W-MON")
        monthly_metrics = self.compute_rollup_metrics(daily_eval_frame, "MS")

        print(f"Best params from RandomizedSearchCV: {self.best_params}")
        self.print_metrics("Daily", daily_metrics)
        self.print_metrics("Weekly", weekly_metrics)
        self.print_metrics("Monthly", monthly_metrics)

        self.save_model()

    def load_data(self) -> None:
        """Load source CSV and validate expected schema."""
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
            raise ValueError("Input dataset is empty after preprocessing.")

        self.df_raw = df

    def filter_top_sellers(self) -> None:
        """Keep top-N items by total lifetime quantity to reduce sparse-item noise."""
        self._ensure(self.df_raw, "Raw data is not loaded. Call load_data() first.")

        totals = (
            self.df_raw.groupby(self.product_col, as_index=False)[self.target_col]
            .sum()
            .sort_values(self.target_col, ascending=False)
        )
        top_items = totals.head(self.top_n_items)[self.product_col]
        self.df_raw = self.df_raw[self.df_raw[self.product_col].isin(top_items)].copy()

        if self.df_raw.empty:
            raise ValueError("No rows left after top-item filtering.")

    def aggregate_daily_with_calendar_fill(self) -> None:
        """Aggregate to daily item level and fill missing item-date rows."""
        self._ensure(self.df_raw, "Raw data is not loaded. Call load_data() first.")

        aggregated = (
            self.df_raw.groupby([self.product_col, self.date_col], as_index=False)
            .agg(
                Quantity=(self.target_col, "sum"),
                SellingPrice=(self.price_col, "mean"),
            )
            .sort_values([self.product_col, self.date_col])
            .reset_index(drop=True)
        )

        all_items = aggregated[self.product_col].unique()
        min_date = aggregated[self.date_col].min()
        max_date = aggregated[self.date_col].max()
        full_dates = pd.date_range(start=min_date, end=max_date, freq="D")

        full_index = pd.MultiIndex.from_product(
            [all_items, full_dates], names=[self.product_col, self.date_col]
        )

        daily = (
            aggregated.set_index([self.product_col, self.date_col])
            .reindex(full_index)
            .reset_index()
        )

        daily[self.target_col] = daily[self.target_col].fillna(0.0)
        # For generated no-sale days, carry nearest known price per item.
        daily[self.price_col] = (
            daily.groupby(self.product_col)[self.price_col]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0.0)
        )

        self.df_daily = daily.sort_values([self.product_col, self.date_col]).reset_index(drop=True)

    def create_daily_features(self) -> None:
        """Create lag, rolling, EWMA, and calendar features from daily base series."""
        self._ensure(
            self.df_daily,
            "Daily data is not prepared. Call aggregate_daily_with_calendar_fill() first.",
        )

        df = self.df_daily.copy()
        df = df.sort_values([self.product_col, self.date_col]).reset_index(drop=True)
        grouped = df.groupby(self.product_col)

        df["lag_1"] = grouped[self.target_col].shift(1)
        df["lag_2"] = grouped[self.target_col].shift(2)
        df["lag_3"] = grouped[self.target_col].shift(3)
        df["lag_7"] = grouped[self.target_col].shift(7)
        df["lag_14"] = grouped[self.target_col].shift(14)
        df["lag_21"] = grouped[self.target_col].shift(21)
        df["lag_28"] = grouped[self.target_col].shift(28)

        df["rolling_mean_7"] = grouped[self.target_col].transform(
            lambda s: s.shift(1).rolling(window=7, min_periods=7).mean()
        )
        df["rolling_mean_28"] = grouped[self.target_col].transform(
            lambda s: s.shift(1).rolling(window=28, min_periods=28).mean()
        )

        df["ewma_7"] = grouped[self.target_col].transform(
            lambda s: s.shift(1).ewm(span=7, adjust=False).mean()
        )
        df["ewma_14"] = grouped[self.target_col].transform(
            lambda s: s.shift(1).ewm(span=14, adjust=False).mean()
        )

        df["DayOfWeek"] = df[self.date_col].dt.dayofweek.astype(int)
        df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["Month"] = df[self.date_col].dt.month.astype(int)
        df["Is_Month_Start"] = df[self.date_col].dt.is_month_start.astype(int)
        df["Is_Month_End"] = df[self.date_col].dt.is_month_end.astype(int)

        required_feature_cols = [
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_7",
            "lag_14",
            "lag_21",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_28",
            "ewma_7",
            "ewma_14",
        ]
        df = df.dropna(subset=required_feature_cols).reset_index(drop=True)

        if df.empty:
            raise ValueError("No rows left after feature engineering. Check data volume and lag windows.")

        self.df_features = df

    def apply_log_transform_and_encoding(self) -> None:
        """Apply log transform to target and one-hot encode item identity."""
        self._ensure(
            self.df_features,
            "Feature data is unavailable. Call create_daily_features() first.",
        )

        df = self.df_features.copy()
        if (df[self.target_col] < 0).any():
            raise ValueError("Quantity has negative values; log1p requires non-negative targets.")

        df[self.log_target_col] = np.log1p(df[self.target_col])

        item_dummies = pd.get_dummies(df[self.product_col], prefix="item", dtype=int)
        df = pd.concat([df, item_dummies], axis=1)

        self.feature_columns = [
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_7",
            "lag_14",
            "lag_21",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_28",
            "ewma_7",
            "ewma_14",
            "SellingPrice",
            "DayOfWeek",
            "Is_Weekend",
            "Month",
            "Is_Month_Start",
            "Is_Month_End",
            *item_dummies.columns.tolist(),
        ]

        self.df_features = df

    def chronological_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """Strict chronological split (80/20) for training and testing."""
        self._ensure(
            self.df_features,
            "Feature data is unavailable. Call apply_log_transform_and_encoding() first.",
        )

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        df = self.df_features.sort_values(self.date_col).reset_index(drop=True)
        split_idx = int(len(df) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(df) - 1)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Chronological split produced an empty train or test set.")

        X_train = train_df[self.feature_columns]
        y_train = train_df[self.log_target_col]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.log_target_col]
        test_index_frame = test_df[[self.product_col, self.date_col]].reset_index(drop=True)

        return X_train, X_test, y_train, y_test, test_index_frame

    def train_with_randomized_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost with randomized search and time-series cross validation."""
        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=1,
        )

        param_dist = {
            "n_estimators": [300, 500, 700],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [5, 6, 8],
            "subsample": [0.8, 0.9],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=15,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=self.random_state,
            n_jobs=1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params = dict(search.best_params_)

    def predict_and_prepare_daily_eval_frame(
        self,
        X_test: pd.DataFrame,
        y_test_log: pd.Series,
        test_index_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict daily demand and produce evaluation frame on original scale."""
        self._ensure(self.model, "Model is not trained. Call train_with_randomized_search() first.")

        y_pred_log = self.model.predict(X_test)
        y_true_log = y_test_log.to_numpy()

        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_true_log)

        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        y_true = np.clip(y_true, a_min=0.0, a_max=None)

        eval_frame = test_index_frame.copy()
        eval_frame["actual"] = y_true
        eval_frame["predicted"] = y_pred
        return eval_frame

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
        """Compute MAE, RMSE, R2, and safe MAPE."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        non_zero_mask = y_true != 0
        if non_zero_mask.any():
            mape = float(
                mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
                * 100
            )
        else:
            mape = 0.0

        return Metrics(mae=mae, rmse=rmse, r2=r2, mape=mape)

    def compute_rollup_metrics(self, daily_eval_frame: pd.DataFrame, freq: str) -> Metrics:
        """Roll up daily predictions to higher hierarchy and compute metrics."""
        temp = daily_eval_frame.copy()
        temp = temp.sort_values([self.product_col, self.date_col]).reset_index(drop=True)

        rolled = (
            temp.groupby(self.product_col)
            .resample(freq, on=self.date_col)[["actual", "predicted"]]
            .sum()
            .reset_index()
        )

        return self.compute_metrics(
            rolled["actual"].to_numpy(),
            rolled["predicted"].to_numpy(),
        )

    @staticmethod
    def print_metrics(level_name: str, metrics: Metrics) -> None:
        print(f"\n{level_name} Metrics")
        print(f"MAE       : {metrics.mae:.4f}")
        print(f"RMSE      : {metrics.rmse:.4f}")
        print(f"R-squared : {metrics.r2:.4f}")
        print(f"MAPE      : {metrics.mape:.4f}%")

    def save_model(self) -> None:
        """Save champion daily model artifact."""
        self._ensure(self.model, "No trained model found to save.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.output_path)
        print(f"\nModel saved to: {self.output_path}")

    @staticmethod
    def _ensure(value: object, message: str) -> None:
        if value is None:
            raise ValueError(message)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "processed" / "cleaned_pos_data.csv"
    output_model_file = project_root / "models" / "xgboost_champion.pkl"

    forecaster = AdvancedHierarchicalForecaster(
        input_path=input_file,
        output_path=output_model_file,
        top_n_items=50,
        train_ratio=0.8,
        random_state=42,
    )
    forecaster.run()


if __name__ == "__main__":
    main()
