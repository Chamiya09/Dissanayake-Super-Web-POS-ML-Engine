from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("xgboost is required. Install with: pip install xgboost") from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with: pip install lightgbm") from exc


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


@dataclass
class Metrics:
    level: str
    r2: float
    mae: float
    rmse: float
    mape: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 6 - Ultimate optimizer for daily/weekly/monthly retail forecasting"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/final_features.csv"),
        help="Input feature dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/demand_forecast_ensemble.pkl"),
        help="Output model artifact",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("exports/plots/step6_top_item_forecast_vs_actual.png"),
        help="Forecast vs Actual plot output path",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for dropping redundant features",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


class UltimateOptimizer:
    """Production-oriented training pipeline with robust optimization and reporting."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        plot_output: Path,
        train_ratio: float,
        corr_threshold: float,
        random_state: int,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.plot_output = plot_output
        self.train_ratio = train_ratio
        self.corr_threshold = corr_threshold
        self.random_state = random_state

        self.date_col = "Date"
        self.id_col = "ProductID"
        self.name_col = "ProductName"
        self.qty_col = "Quantity"
        self.target_col = "log_quantity"

        self.keep_priority_cols = {
            "Is_Avurudu_Season",
            "Is_Vesak_Season",
            "Is_Christmas_Season",
        }

    def run(self) -> None:
        df = self.load_data()
        df = self.clip_target_outliers(df)

        split = self.chronological_split(df)
        train_df, test_df = split

        feature_cols = self.get_base_feature_columns(train_df)
        feature_cols, dropped_corr_cols = self.drop_redundant_features(train_df, feature_cols)

        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        y_train = train_df[self.target_col]
        y_test = test_df[self.target_col]

        scaled = self.apply_standardization(X_train, X_test)
        X_train_scaled, X_test_scaled, scaler, scaled_cols = scaled

        xgb_model, xgb_objective = self.select_best_xgb_objective(X_train_scaled, y_train)
        lgbm_model, lgbm_objective = self.select_best_lgbm_objective(X_train_scaled, y_train)
        rf_model = self.build_rf_model()

        ensemble = VotingRegressor(
            estimators=[("xgb", xgb_model), ("lgbm", lgbm_model), ("rf", rf_model)],
            weights=[0.4, 0.4, 0.2],
        )
        ensemble.fit(X_train_scaled, y_train)

        daily_results = self.predict_with_metadata(
            model=ensemble,
            X_test=X_test_scaled,
            y_test_log=y_test,
            test_df=test_df,
        )

        daily_metrics = self.compute_metrics(
            daily_results["Actual_Qty"].to_numpy(),
            daily_results["Predicted_Qty"].to_numpy(),
            "Daily",
        )

        weekly_df = self.roll_up(daily_results, "W-MON")
        weekly_metrics = self.compute_metrics(
            weekly_df["Actual_Qty"].to_numpy(),
            weekly_df["Predicted_Qty"].to_numpy(),
            "Weekly",
        )

        monthly_df = self.roll_up(daily_results, "MS")
        monthly_metrics = self.compute_metrics(
            monthly_df["Actual_Qty"].to_numpy(),
            monthly_df["Predicted_Qty"].to_numpy(),
            "Monthly",
        )

        self.print_metrics_table([daily_metrics, weekly_metrics, monthly_metrics])
        self.plot_top_item_forecast(daily_results)

        artifact: dict[str, Any] = {
            "model": ensemble,
            "feature_columns": feature_cols,
            "scaled_columns": scaled_cols,
            "scaler": scaler,
            "dropped_corr_columns": dropped_corr_cols,
            "xgb_objective": xgb_objective,
            "lgbm_objective": lgbm_objective,
            "target_column": self.target_col,
            "inverse_transform": "np.expm1",
            "weights": {"xgb": 0.4, "lgbm": 0.4, "rf": 0.2},
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.output_path)
        print(f"\nSaved optimized ensemble artifact to: {self.output_path}")

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        required = {self.date_col, self.id_col, self.qty_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.qty_col] = pd.to_numeric(df[self.qty_col], errors="coerce")

        df = df.dropna(subset=[self.date_col, self.id_col, self.qty_col]).copy()
        if df.empty:
            raise ValueError("Dataset is empty after parsing and null removal.")

        return df

    def clip_target_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        p95 = out[self.qty_col].quantile(0.95)
        out[self.qty_col] = np.where(out[self.qty_col] > p95, p95, out[self.qty_col])
        out[self.target_col] = np.log1p(np.clip(out[self.qty_col], a_min=0.0, a_max=None))
        return out

    def chronological_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not 0 < self.train_ratio < 1:
            raise ValueError("train-ratio must be between 0 and 1.")

        ordered = df.sort_values(self.date_col).reset_index(drop=True)
        split_idx = int(len(ordered) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(ordered) - 1)
        train_df = ordered.iloc[:split_idx].copy()
        test_df = ordered.iloc[split_idx:].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Train/test split produced empty dataframes.")

        return train_df, test_df

    def get_base_feature_columns(self, train_df: pd.DataFrame) -> list[str]:
        exclude_cols = {
            self.date_col,
            self.id_col,
            self.name_col,
            self.qty_col,
            self.target_col,
        }
        candidate_cols = [c for c in train_df.columns if c not in exclude_cols]
        numeric_cols = train_df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric feature columns available for training.")
        return numeric_cols

    def drop_redundant_features(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        feature_frame = train_df[feature_cols]
        corr = feature_frame.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop: list[str] = []
        for col in upper.columns:
            if col in self.keep_priority_cols:
                continue
            if (upper[col] > self.corr_threshold).any():
                to_drop.append(col)

        kept = [c for c in feature_cols if c not in set(to_drop)]

        # Ensure priority holiday flags are retained even if highly correlated.
        for priority_col in self.keep_priority_cols:
            if priority_col in feature_cols and priority_col not in kept:
                kept.append(priority_col)

        return kept, sorted(set(to_drop))

    def apply_standardization(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
        train = X_train.copy()
        test = X_test.copy()

        numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
        # Avoid scaling binary one-hot/flags; only scale columns with >2 distinct values.
        scale_cols = [c for c in numeric_cols if train[c].nunique(dropna=False) > 2]

        scaler = StandardScaler()
        if scale_cols:
            train[scale_cols] = scaler.fit_transform(train[scale_cols])
            test[scale_cols] = scaler.transform(test[scale_cols])
        else:
            scaler.fit(np.zeros((len(train), 1)))

        return train, test, scaler, scale_cols

    def _inner_split(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        split_idx = int(len(X) * 0.8)
        split_idx = min(max(split_idx, 1), len(X) - 1)

        X_tr = X.iloc[:split_idx].copy()
        X_val = X.iloc[split_idx:].copy()
        y_tr = y.iloc[:split_idx].copy()
        y_val = y.iloc[split_idx:].copy()
        return X_tr, X_val, y_tr, y_val

    def select_best_xgb_objective(self, X: pd.DataFrame, y: pd.Series) -> tuple[XGBRegressor, str]:
        X_tr, X_val, y_tr, y_val = self._inner_split(X, y)

        candidates = ["reg:absoluteerror", "reg:squarederror"]
        best_model: XGBRegressor | None = None
        best_obj = ""
        best_score = float("inf")

        for obj in candidates:
            model = XGBRegressor(
                objective=obj,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=1,
            )
            model.fit(X_tr, y_tr)
            pred = np.expm1(model.predict(X_val))
            true = np.expm1(y_val.to_numpy())

            rmse = np.sqrt(mean_squared_error(true, pred))
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_obj = obj

        assert best_model is not None
        best_model.fit(X, y)
        print(f"Selected XGBoost objective: {best_obj} | validation RMSE={best_score:.4f}")
        return best_model, best_obj

    def select_best_lgbm_objective(self, X: pd.DataFrame, y: pd.Series) -> tuple[LGBMRegressor, str]:
        X_tr, X_val, y_tr, y_val = self._inner_split(X, y)

        obj_map = {
            "reg:absoluteerror": "regression_l1",
            "reg:squarederror": "regression",
        }

        best_model: LGBMRegressor | None = None
        best_obj = ""
        best_score = float("inf")

        for label, lgb_obj in obj_map.items():
            model = LGBMRegressor(
                objective=lgb_obj,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_tr, y_tr)
            pred = np.expm1(model.predict(X_val))
            true = np.expm1(y_val.to_numpy())

            rmse = np.sqrt(mean_squared_error(true, pred))
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_obj = label

        assert best_model is not None
        best_model.fit(X, y)
        print(f"Selected LightGBM objective: {best_obj} | validation RMSE={best_score:.4f}")
        return best_model, best_obj

    def build_rf_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def predict_with_metadata(
        self,
        model: VotingRegressor,
        X_test: pd.DataFrame,
        y_test_log: pd.Series,
        test_df: pd.DataFrame,
    ) -> pd.DataFrame:
        pred_log = model.predict(X_test)
        true_log = y_test_log.to_numpy()

        pred_qty = np.expm1(pred_log)
        true_qty = np.expm1(true_log)

        pred_qty = np.clip(pred_qty, a_min=0.0, a_max=None)
        true_qty = np.clip(true_qty, a_min=0.0, a_max=None)

        out = test_df[[self.date_col, self.id_col, self.name_col]].copy()
        out["Actual_Qty"] = true_qty
        out["Predicted_Qty"] = pred_qty
        return out

    def roll_up(self, results: pd.DataFrame, freq: str) -> pd.DataFrame:
        temp = results.copy()
        temp[self.date_col] = pd.to_datetime(temp[self.date_col], errors="coerce")

        rolled = (
            temp.groupby(self.id_col)
            .resample(freq, on=self.date_col)[["Actual_Qty", "Predicted_Qty"]]
            .sum()
            .reset_index()
        )
        return rolled

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, level: str) -> Metrics:
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        nz = y_true != 0
        if nz.any():
            mape = float(mean_absolute_percentage_error(y_true[nz], y_pred[nz]) * 100)
        else:
            mape = 0.0

        return Metrics(level=level, r2=r2, mae=mae, rmse=rmse, mape=mape)

    def print_metrics_table(self, rows: list[Metrics]) -> None:
        print("\nAdvanced Metrics Table")
        print(f"{'Level':<10}{'R2':>10}{'MAE':>12}{'RMSE':>12}{'MAPE(%)':>12}{'Target Check':>18}")

        for row in rows:
            if row.level == "Monthly":
                passed = row.mape < 15
            elif row.level == "Weekly":
                passed = row.mape < 25
            else:
                passed = False

            if row.level in {"Weekly", "Monthly"}:
                check = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
            else:
                check = "-"

            print(
                f"{row.level:<10}{row.r2:>10.4f}{row.mae:>12.4f}{row.rmse:>12.4f}{row.mape:>12.4f}{check:>18}"
            )

    def plot_top_item_forecast(self, daily_results: pd.DataFrame) -> None:
        if daily_results.empty:
            print("[WARN] No daily results available for plotting.")
            return

        top_item = (
            daily_results.groupby(self.id_col, as_index=False)["Actual_Qty"]
            .sum()
            .sort_values("Actual_Qty", ascending=False)
            .iloc[0][self.id_col]
        )

        item_df = daily_results[daily_results[self.id_col] == top_item].copy()
        item_df = item_df.sort_values(self.date_col)

        if item_df.empty:
            print("[WARN] Top item subset for plotting is empty.")
            return

        self.plot_output.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 6))
        plt.plot(item_df[self.date_col], item_df["Actual_Qty"], label="Actual", linewidth=2)
        plt.plot(item_df[self.date_col], item_df["Predicted_Qty"], label="Predicted", linewidth=2)
        plt.title(f"Forecast vs Actual - Top Selling Item ({top_item})")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_output)
        plt.close()

        print(f"Saved visual validation plot to: {self.plot_output}")


def main() -> None:
    args = parse_args()
    optimizer = UltimateOptimizer(
        input_path=args.input,
        output_path=args.output,
        plot_output=args.plot_output,
        train_ratio=args.train_ratio,
        corr_threshold=args.corr_threshold,
        random_state=args.random_state,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
