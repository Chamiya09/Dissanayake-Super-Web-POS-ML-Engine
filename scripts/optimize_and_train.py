from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install with: pip install xgboost"
    ) from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "lightgbm is required. Install with: pip install lightgbm"
    ) from exc


@dataclass
class MetricRow:
    level: str
    segment: str
    r2: float
    rmse: float
    mae: float
    mape: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4 - Hyperparameter Optimization and Error Analysis"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/final_features.csv"),
        help="Path to engineered feature dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/demand_forecast_ensemble.pkl"),
        help="Path to save optimized ensemble artifact",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological train ratio",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help="Random search iterations per model",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


class OptimizedEnsembleTrainer:
    """Optimize base models and train weighted ensemble with tier-aware analysis."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        train_ratio: float,
        n_iter: int,
        random_state: int,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.train_ratio = train_ratio
        self.n_iter = n_iter
        self.random_state = random_state

        self.date_col = "Date"
        self.id_col = "ProductID"
        self.target_col = "log_quantity"
        self.raw_target_col = "Quantity"

    def run(self) -> None:
        df = self.load_data()
        df = self.cap_outliers(df)

        prepared = self.prepare_split(df)
        X_train, X_test, y_train, y_test, test_meta = prepared

        xgb_best, xgb_rmse = self.tune_xgboost(X_train, y_train)
        lgbm_best, lgbm_rmse = self.tune_lightgbm(X_train, y_train)
        rf_model, rf_rmse = self.train_random_forest_baseline(X_train, y_train)

        weights = self.build_weights(
            xgb_rmse=xgb_rmse,
            lgbm_rmse=lgbm_rmse,
            rf_rmse=rf_rmse,
        )

        ensemble = VotingRegressor(
            estimators=[("xgb", xgb_best), ("lgbm", lgbm_best), ("rf", rf_model)],
            weights=weights,
        )
        ensemble.fit(X_train, y_train)

        daily_results = self.predict_with_metadata(ensemble, X_test, y_test, test_meta)
        tier_map = self.create_tier_map(df)

        metrics = []
        metrics.extend(self.evaluate_overall_and_tier(daily_results, tier_map, level_name="Daily"))

        weekly_results = self.rollup(daily_results, "W-MON")
        metrics.extend(self.evaluate_overall_and_tier(weekly_results, tier_map, level_name="Weekly"))

        monthly_results = self.rollup(daily_results, "MS")
        metrics.extend(self.evaluate_overall_and_tier(monthly_results, tier_map, level_name="Monthly"))

        self.print_metric_table(metrics)

        artifact: dict[str, Any] = {
            "model": ensemble,
            "feature_columns": X_train.columns.tolist(),
            "weights": {
                "xgb": weights[0],
                "lgbm": weights[1],
                "rf": weights[2],
            },
            "xgb_best_params": xgb_best.get_params(),
            "lgbm_best_params": lgbm_best.get_params(),
            "rf_params": rf_model.get_params(),
            "target_column": self.target_col,
            "inverse_transform": "np.expm1",
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.output_path)
        print(f"\nSaved optimized model artifact to: {self.output_path}")

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        required = {self.date_col, self.id_col, self.raw_target_col, self.target_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.raw_target_col] = pd.to_numeric(df[self.raw_target_col], errors="coerce")

        df = df.dropna(subset=[self.date_col, self.id_col, self.raw_target_col]).copy()
        if df.empty:
            raise ValueError("Dataset empty after basic cleanup.")

        return df

    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap quantity-driven features by ProductID at 99th percentile."""
        out = df.copy()

        quantity_like_cols = [
            c
            for c in [
                "Quantity",
                "lag_1",
                "lag_2",
                "lag_3",
                "lag_7",
                "lag_14",
                "lag_30",
                "rolling_mean_7",
                "rolling_mean_30",
                "rolling_std_7",
            ]
            if c in out.columns
        ]

        for col in quantity_like_cols:
            caps = out.groupby(self.id_col)[col].transform(lambda s: s.quantile(0.99))
            out[col] = np.where(out[col] > caps, caps, out[col])

        if (out[self.raw_target_col] < 0).any():
            raise ValueError("Negative Quantity values found after outlier handling.")

        out[self.target_col] = np.log1p(out[self.raw_target_col])
        return out

    def prepare_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        if not 0 < self.train_ratio < 1:
            raise ValueError("train-ratio must be between 0 and 1.")

        ordered = df.sort_values(self.date_col).reset_index(drop=True)
        split_idx = int(len(ordered) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(ordered) - 1)

        train_df = ordered.iloc[:split_idx].copy()
        test_df = ordered.iloc[split_idx:].copy()

        exclude_cols = {
            self.date_col,
            self.id_col,
            self.target_col,
            self.raw_target_col,
            "ProductName",
        }
        candidate_cols = [c for c in train_df.columns if c not in exclude_cols]
        feature_cols = train_df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

        # LightGBM warns when feature names contain whitespace; normalize once here.
        sanitized_feature_cols = [c.replace(" ", "_") for c in feature_cols]
        rename_map = dict(zip(feature_cols, sanitized_feature_cols))

        X_train = train_df[feature_cols].rename(columns=rename_map)
        X_test = test_df[feature_cols].rename(columns=rename_map)
        y_train = train_df[self.target_col]
        y_test = test_df[self.target_col]
        test_meta = test_df[[self.date_col, self.id_col]].copy()

        return X_train, X_test, y_train, y_test, test_meta

    def tune_xgboost(self, X: pd.DataFrame, y: pd.Series) -> tuple[XGBRegressor, float]:
        base = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=1,
        )

        param_dist = {
            "max_depth": [4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=self.random_state,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X, y)

        best_rmse = -float(search.best_score_)
        print(f"Best XGBoost RMSE (CV): {best_rmse:.5f}")
        return search.best_estimator_, best_rmse

    def tune_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> tuple[LGBMRegressor, float]:
        base = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        param_dist = {
            "max_depth": [4, 5, 6, 8, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=self.random_state,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X, y)

        best_model: LGBMRegressor = search.best_estimator_

        best_rmse = -float(search.best_score_)
        print(f"Best LightGBM RMSE (CV): {best_rmse:.5f}")
        return best_model, best_rmse

    def train_random_forest_baseline(self, X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestRegressor, float]:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

        tscv = TimeSeriesSplit(n_splits=3)
        rmse_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, pred)))

        model.fit(X, y)
        avg_rmse = float(np.mean(rmse_scores))
        print(f"RandomForest RMSE (CV avg): {avg_rmse:.5f}")
        return model, avg_rmse

    @staticmethod
    def build_weights(xgb_rmse: float, lgbm_rmse: float, rf_rmse: float) -> list[float]:
        rmses = np.array([xgb_rmse, lgbm_rmse, rf_rmse], dtype=float)
        inv = 1.0 / np.clip(rmses, 1e-9, None)
        weights = inv / inv.sum()
        return weights.tolist()

    def predict_with_metadata(
        self,
        model: VotingRegressor,
        X_test: pd.DataFrame,
        y_test_log: pd.Series,
        test_meta: pd.DataFrame,
    ) -> pd.DataFrame:
        # VotingRegressor passes original feature names to each model;
        # LightGBM warnings are avoided by ensuring sanitized columns in wrapper fit/predict.
        # To keep interoperability, use original columns here and clip inverse outputs.
        pred_log = model.predict(X_test)
        true_log = y_test_log.to_numpy()

        pred_qty = np.expm1(pred_log)
        true_qty = np.expm1(true_log)

        pred_qty = np.clip(pred_qty, a_min=0.0, a_max=None)
        true_qty = np.clip(true_qty, a_min=0.0, a_max=None)

        out = test_meta.copy()
        out["Actual_Qty"] = true_qty
        out["Predicted_Qty"] = pred_qty
        return out

    def create_tier_map(self, df: pd.DataFrame) -> pd.DataFrame:
        totals = (
            df.groupby(self.id_col, as_index=False)[self.raw_target_col]
            .sum()
            .rename(columns={self.raw_target_col: "total_qty"})
        )
        q33 = totals["total_qty"].quantile(0.33)
        q67 = totals["total_qty"].quantile(0.67)

        totals["tier"] = np.where(
            totals["total_qty"] >= q67,
            "Tier 1",
            np.where(totals["total_qty"] <= q33, "Tier 3", "Tier 2"),
        )
        return totals[[self.id_col, "tier"]]

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

    def evaluate_overall_and_tier(
        self,
        frame: pd.DataFrame,
        tier_map: pd.DataFrame,
        level_name: str,
    ) -> list[MetricRow]:
        merged = frame.merge(tier_map, on=self.id_col, how="left")
        rows = [self.compute_metric_row(merged, level_name, "Overall")]
        rows.append(self.compute_metric_row(merged[merged["tier"] == "Tier 1"], level_name, "Tier 1"))
        rows.append(self.compute_metric_row(merged[merged["tier"] == "Tier 3"], level_name, "Tier 3"))
        return rows

    @staticmethod
    def compute_metric_row(frame: pd.DataFrame, level: str, segment: str) -> MetricRow:
        if frame.empty:
            return MetricRow(level=level, segment=segment, r2=np.nan, rmse=np.nan, mae=np.nan, mape=np.nan)

        y_true = frame["Actual_Qty"].to_numpy()
        y_pred = frame["Predicted_Qty"].to_numpy()

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        nz = y_true != 0
        if nz.any():
            mape = float(mean_absolute_percentage_error(y_true[nz], y_pred[nz]) * 100)
        else:
            mape = 0.0

        return MetricRow(level=level, segment=segment, r2=r2, rmse=rmse, mae=mae, mape=mape)

    @staticmethod
    def print_metric_table(rows: list[MetricRow]) -> None:
        df = pd.DataFrame(
            {
                "Level": [r.level for r in rows],
                "Segment": [r.segment for r in rows],
                "R2": [r.r2 for r in rows],
                "RMSE": [r.rmse for r in rows],
                "MAE": [r.mae for r in rows],
                "MAPE(%)": [r.mape for r in rows],
            }
        )
        print("\nOptimized Accuracy Table")
        print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = parse_args()

    trainer = OptimizedEnsembleTrainer(
        input_path=args.input,
        output_path=args.output,
        train_ratio=args.train_ratio,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )
    trainer.run()


if __name__ == "__main__":
    main()
