from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
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
class MetricRow:
    level: str
    r2: float
    mae: float
    rmse: float
    mape: float


class FinalProductionModel:
    """Integrated Step 5 & 6 production-grade retail demand forecasting pipeline."""

    def __init__(self) -> None:
        self.input_path = Path("data/processed/final_features.csv")
        self.model_path = Path("models/final_production_model.pkl")
        self.plot_path = Path("exports/plots/final_production_top_item_forecast_vs_actual.png")

        self.date_col = "Date"
        self.id_col = "ProductID"
        self.name_col = "ProductName"
        self.qty_col = "Quantity"
        self.log_qty_col = "log_quantity"

        self.holiday_flags = [
            "Is_Avurudu_Season",
            "Is_Vesak_Season",
            "Is_Christmas_Season",
        ]

        self.random_state = 42
        self.train_ratio = 0.8

    def run(self) -> None:
        df = self.load_data()
        df = self.add_event_distance_features(df)
        df = self.add_category_intelligence(df)
        df = self.clip_target_outliers(df)

        train_df, test_df = self.time_split(df)
        feature_cols = self.select_feature_columns(train_df)
        feature_cols, dropped_cols = self.apply_correlation_filter(train_df, feature_cols)

        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        y_train_qty = train_df[self.qty_col].to_numpy()
        y_test_qty = test_df[self.qty_col].to_numpy()

        y_train_occ = (y_train_qty > 0).astype(int)
        y_test_occ = (y_test_qty > 0).astype(int)

        X_train_scaled, X_test_scaled, scaler, scaled_cols = self.scale_features(X_train, X_test)

        clf = self.train_stage1_classifier(X_train_scaled, y_train_occ)
        reg = self.train_stage2_regressor(X_train_scaled, y_train_qty)

        pred_daily_df = self.predict_daily(
            clf=clf,
            reg=reg,
            X_test_scaled=X_test_scaled,
            y_test_qty=y_test_qty,
            test_df=test_df,
            feature_cols=feature_cols,
            scaled_cols=scaled_cols,
        )

        daily_metrics = self.compute_metrics(
            pred_daily_df["Actual_Qty"].to_numpy(),
            pred_daily_df["Predicted_Qty"].to_numpy(),
            "Daily",
        )

        weekly_df = self.roll_up(pred_daily_df, "W-MON")
        weekly_metrics = self.compute_metrics(
            weekly_df["Actual_Qty"].to_numpy(),
            weekly_df["Predicted_Qty"].to_numpy(),
            "Weekly",
        )

        monthly_df = self.roll_up(pred_daily_df, "MS")
        monthly_metrics = self.compute_metrics(
            monthly_df["Actual_Qty"].to_numpy(),
            monthly_df["Predicted_Qty"].to_numpy(),
            "Monthly",
        )

        self.print_metrics_table([daily_metrics, weekly_metrics, monthly_metrics])
        self.plot_top_item(pred_daily_df)

        artifact: dict[str, Any] = {
            "stage1_classifier": clf,
            "stage2_stacking_regressor": reg,
            "feature_columns": feature_cols,
            "scaled_columns": scaled_cols,
            "scaler": scaler,
            "dropped_corr_columns": dropped_cols,
            "holiday_flags": self.holiday_flags,
            "target_transform": "log1p",
            "inverse_transform": "expm1",
        }

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.model_path)
        print(f"\nSaved final production model to: {self.model_path}")

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input feature file not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        required = {self.date_col, self.id_col, self.qty_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.qty_col] = pd.to_numeric(df[self.qty_col], errors="coerce")
        if self.log_qty_col in df.columns:
            df[self.log_qty_col] = pd.to_numeric(df[self.log_qty_col], errors="coerce")

        df = df.dropna(subset=[self.date_col, self.id_col, self.qty_col]).copy()
        if df.empty:
            raise ValueError("Input dataset is empty after basic parsing.")

        return df

    def add_event_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Avurudu anchor date every year: April 13.
        avurudu_anchor = pd.to_datetime(out[self.date_col].dt.year.astype(str) + "-04-13")
        out["days_until_avurudu"] = (avurudu_anchor - out[self.date_col]).dt.days

        # Payday anchor: 25th each month.
        payday_anchor = pd.to_datetime(
            out[self.date_col].dt.year.astype(str)
            + "-"
            + out[self.date_col].dt.month.astype(str).str.zfill(2)
            + "-25"
        )
        out["days_until_payday"] = (payday_anchor - out[self.date_col]).dt.days

        # Poya approximation: nearest lunar full-moon cycle from fixed anchor (2019-01-21).
        lunar_anchor = pd.Timestamp("2019-01-21")
        days_since = (out[self.date_col] - lunar_anchor).dt.days.astype(float)
        cycle = 29.530588
        k = np.round(days_since / cycle)
        next_poya = lunar_anchor + pd.to_timedelta(k * cycle, unit="D")
        out["days_until_poya"] = (next_poya - out[self.date_col]).dt.days

        return out

    def add_category_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        category_cols = [c for c in out.columns if c.startswith("Category_")]
        if category_cols:
            out["_category_label"] = out[category_cols].idxmax(axis=1).str.replace("Category_", "", regex=False)
        else:
            out["_category_label"] = "Unknown"

        out["_daily_category_qty"] = out.groupby([self.date_col, "_category_label"])[self.qty_col].transform("mean")

        out = out.sort_values(["_category_label", self.date_col]).reset_index(drop=True)
        out["category_rolling_mean_7"] = (
            out.groupby("_category_label")["_daily_category_qty"]
            .transform(lambda s: s.shift(1).rolling(window=7, min_periods=7).mean())
        )

        # Keep continuity in category context.
        out["category_rolling_mean_7"] = out["category_rolling_mean_7"].fillna(out["_daily_category_qty"])

        out = out.drop(columns=["_daily_category_qty", "_category_label"])
        return out

    def clip_target_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        p95 = out[self.qty_col].quantile(0.95)
        out[self.qty_col] = np.where(out[self.qty_col] > p95, p95, out[self.qty_col])
        out[self.qty_col] = np.clip(out[self.qty_col], a_min=0.0, a_max=None)
        out[self.log_qty_col] = np.log1p(out[self.qty_col])
        return out

    def time_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = df.sort_values(self.date_col).reset_index(drop=True)
        split_idx = int(len(ordered) * self.train_ratio)
        split_idx = min(max(split_idx, 1), len(ordered) - 1)

        train_df = ordered.iloc[:split_idx].copy()
        test_df = ordered.iloc[split_idx:].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Time split failed: train/test empty.")

        return train_df, test_df

    def select_feature_columns(self, train_df: pd.DataFrame) -> list[str]:
        exclude_cols = {
            self.date_col,
            self.id_col,
            self.name_col,
            self.qty_col,
            self.log_qty_col,
        }

        candidate = [c for c in train_df.columns if c not in exclude_cols]
        numeric = train_df[candidate].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric:
            raise ValueError("No numeric features found for training.")
        return numeric

    def apply_correlation_filter(self, train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
        corr = train_df[feature_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop: list[str] = []
        for col in upper.columns:
            if col in self.holiday_flags:
                continue
            if (upper[col] > 0.9).any():
                to_drop.append(col)

        kept = [c for c in feature_cols if c not in set(to_drop)]
        for holiday in self.holiday_flags:
            if holiday in feature_cols and holiday not in kept:
                kept.append(holiday)

        return kept, sorted(set(to_drop))

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
        train = X_train.copy()
        test = X_test.copy()

        num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
        scale_cols = [c for c in num_cols if train[c].nunique(dropna=False) > 2]

        scaler = StandardScaler()
        if scale_cols:
            train[scale_cols] = scaler.fit_transform(train[scale_cols])
            test[scale_cols] = scaler.transform(test[scale_cols])
        else:
            scaler.fit(np.zeros((len(train), 1)))

        return train, test, scaler, scale_cols

    def train_stage1_classifier(self, X: pd.DataFrame, y_binary: np.ndarray) -> RandomForestClassifier:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X, y_binary)
        return clf

    def train_stage2_regressor(self, X: pd.DataFrame, y_qty: np.ndarray) -> StackingRegressor:
        positive_mask = y_qty > 0
        X_pos = X.loc[positive_mask].copy()
        y_pos_log = np.log1p(y_qty[positive_mask])

        if X_pos.empty:
            raise ValueError("No positive-quantity rows available for stage-2 regressor training.")

        xgb = XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            n_jobs=1,
        )

        lgbm = LGBMRegressor(
            objective="regression_l1",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        rf = RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1,
        )

        stack = StackingRegressor(
            estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
            final_estimator=LinearRegression(),
            cv=5,
            passthrough=False,
            n_jobs=1,
        )
        stack.fit(X_pos, y_pos_log)
        return stack

    def predict_daily(
        self,
        clf: RandomForestClassifier,
        reg: StackingRegressor,
        X_test_scaled: pd.DataFrame,
        y_test_qty: np.ndarray,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        scaled_cols: list[str],
    ) -> pd.DataFrame:
        demand_pred = clf.predict(X_test_scaled)

        pred_qty = np.zeros(len(X_test_scaled), dtype=float)
        positive_idx = np.where(demand_pred == 1)[0]

        if len(positive_idx) > 0:
            reg_pred_log = reg.predict(X_test_scaled.iloc[positive_idx])
            reg_pred_qty = np.expm1(reg_pred_log)
            pred_qty[positive_idx] = np.clip(reg_pred_qty, a_min=0.0, a_max=None)

        actual_qty = np.clip(y_test_qty, a_min=0.0, a_max=None)

        out = test_df[[self.date_col, self.id_col, self.name_col]].copy()
        out["Actual_Qty"] = actual_qty
        out["Predicted_Qty"] = pred_qty
        return out

    def roll_up(self, daily_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        temp = daily_df.copy()
        temp[self.date_col] = pd.to_datetime(temp[self.date_col], errors="coerce")

        return (
            temp.groupby(self.id_col)
            .resample(freq, on=self.date_col)[["Actual_Qty", "Predicted_Qty"]]
            .sum()
            .reset_index()
        )

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, level: str) -> MetricRow:
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        nz = y_true != 0
        if nz.any():
            mape = float(mean_absolute_percentage_error(y_true[nz], y_pred[nz]) * 100)
        else:
            mape = 0.0

        return MetricRow(level=level, r2=r2, mae=mae, rmse=rmse, mape=mape)

    def print_metrics_table(self, rows: list[MetricRow]) -> None:
        print("\nAccuracy Comparison Table")
        print(f"{'Level':<10}{'R2':>10}{'MAE':>12}{'RMSE':>12}{'MAPE(%)':>12}")
        for row in rows:
            print(f"{row.level:<10}{row.r2:>10.4f}{row.mae:>12.4f}{row.rmse:>12.4f}{row.mape:>12.4f}")

        monthly = next((r for r in rows if r.level == "Monthly"), None)
        if monthly is not None:
            if monthly.mape < 15:
                print(f"\n{GREEN}SUCCESS: Monthly MAPE target achieved (<15%).{RESET}")
            else:
                print(f"\n{RED}Monthly MAPE target not met. Current: {monthly.mape:.2f}%{RESET}")

    def plot_top_item(self, daily_df: pd.DataFrame) -> None:
        if daily_df.empty:
            print("[WARN] No daily predictions available to plot.")
            return

        top_item = (
            daily_df.groupby([self.id_col, self.name_col], as_index=False)["Actual_Qty"]
            .sum()
            .sort_values("Actual_Qty", ascending=False)
            .iloc[0]
        )

        item_id = top_item[self.id_col]
        item_name = top_item[self.name_col]

        item_df = daily_df[daily_df[self.id_col] == item_id].copy()
        item_df = item_df.sort_values(self.date_col)

        self.plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 6))
        plt.plot(item_df[self.date_col], item_df["Actual_Qty"], label="Actual", linewidth=2)
        plt.plot(item_df[self.date_col], item_df["Predicted_Qty"], label="Predicted", linewidth=2)
        plt.title(f"Forecast vs Actual - Top Item ({item_id} | {item_name})")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

        print(f"Saved visual validation plot to: {self.plot_path}")


def main() -> None:
    pipeline = FinalProductionModel()
    pipeline.run()


if __name__ == "__main__":
    main()
