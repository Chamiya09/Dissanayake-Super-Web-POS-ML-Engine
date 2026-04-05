from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for retraining. Install it with: pip install xgboost"
    ) from exc


@dataclass
class EvaluationResult:
    champion_rmse: float
    challenger_rmse: float
    promoted: bool


class ChampionChallengerRetrainer:
    """Automated champion-challenger retraining for retail demand forecasting."""

    def __init__(
        self,
        data_path: Path,
        champion_model_path: Path,
        lag_steps: Sequence[int] = (7, 14),
        holdout_days: int = 14,
    ) -> None:
        self.data_path = data_path
        self.champion_model_path = champion_model_path
        self.lag_steps = sorted(set(lag_steps))
        self.holdout_days = holdout_days

        self.date_col = "Date"
        self.item_col = "Item_ID"
        self.target_col = "Quantity"

        self.df_features: pd.DataFrame | None = None
        self.feature_columns: list[str] = []

    def run(self) -> EvaluationResult:
        """Run full champion-challenger loop and return evaluation summary."""
        self.load_and_prepare_features()
        train_df, holdout_df = self.split_train_holdout()

        challenger = self.train_challenger(train_df)
        champion = self.load_champion()

        champion_rmse = self.evaluate_model(champion, holdout_df)
        challenger_rmse = self.evaluate_model(challenger, holdout_df)

        promoted = self.promote_if_better(challenger, challenger_rmse, champion_rmse)

        return EvaluationResult(
            champion_rmse=champion_rmse,
            challenger_rmse=challenger_rmse,
            promoted=promoted,
        )

    def load_and_prepare_features(self) -> None:
        """Load latest cleaned data and create item-level lag features."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        required = {self.date_col, self.target_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        if self.item_col not in df.columns:
            fallbacks = ["Product_ID", "ProductName", "SKU", "ItemId"]
            fallback = next((col for col in fallbacks if col in df.columns), None)
            if fallback is None:
                raise KeyError(
                    "Could not find product identifier column. "
                    f"Checked: {[self.item_col, *fallbacks]}"
                )
            self.item_col = fallback

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col]).copy()

        # Aggregate to daily item-level demand.
        df = (
            df.groupby([self.date_col, self.item_col], as_index=False)[self.target_col]
            .sum()
            .sort_values(self.date_col)
            .reset_index(drop=True)
        )

        for lag in self.lag_steps:
            df[f"lag_{lag}"] = df.groupby(self.item_col)[self.target_col].shift(lag)

        # Calendar features must remain consistent across retraining cycles.
        df["day_of_week"] = df[self.date_col].dt.dayofweek
        df["month"] = df[self.date_col].dt.month

        lag_cols = [f"lag_{lag}" for lag in self.lag_steps]
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No rows left after feature engineering. "
                "Check data volume or reduce lag steps."
            )

        self.feature_columns = lag_cols + ["day_of_week", "month"]
        self.df_features = df

    def split_train_holdout(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split chronologically: train = older data, holdout = most recent 2 weeks."""
        self._ensure(
            self.df_features,
            "Features not prepared. Call load_and_prepare_features() first.",
        )

        df = self.df_features.sort_values(self.date_col).copy()
        max_date = df[self.date_col].max()
        holdout_start = max_date - pd.Timedelta(days=self.holdout_days - 1)

        holdout_df = df[df[self.date_col] >= holdout_start].copy()
        train_df = df[df[self.date_col] < holdout_start].copy()

        if train_df.empty or holdout_df.empty:
            raise ValueError(
                "Invalid time split. Ensure dataset spans enough dates for training and holdout."
            )

        return train_df, holdout_df

    def train_challenger(self, train_df: pd.DataFrame) -> XGBRegressor:
        """Train challenger model using latest available training window."""
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_col]

        challenger = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        challenger.fit(X_train, y_train)
        return challenger

    def load_champion(self):
        """Load current production champion model from disk."""
        if not self.champion_model_path.exists():
            raise FileNotFoundError(
                f"Champion model not found: {self.champion_model_path}"
            )

        try:
            champion = joblib.load(self.champion_model_path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load champion model from {self.champion_model_path}: {exc}"
            ) from exc

        return champion

    def evaluate_model(self, model, holdout_df: pd.DataFrame) -> float:
        """Evaluate model RMSE on holdout data, aligning expected feature order."""
        expected_features = list(getattr(model, "feature_names_in_", []))
        if not expected_features:
            expected_features = self.feature_columns

        missing_features = [col for col in expected_features if col not in holdout_df.columns]
        if missing_features:
            raise ValueError(
                "Holdout data is missing model-required features: "
                f"{missing_features}"
            )

        X_holdout = holdout_df[expected_features]
        y_holdout = holdout_df[self.target_col]

        y_pred = model.predict(X_holdout)
        rmse = float(np.sqrt(mean_squared_error(y_holdout, y_pred)))
        return rmse

    def promote_if_better(
        self,
        challenger,
        challenger_rmse: float,
        champion_rmse: float,
    ) -> bool:
        """Promote challenger to champion only when it outperforms champion."""
        print(f"Champion RMSE   : {champion_rmse:.4f}")
        print(f"Challenger RMSE : {challenger_rmse:.4f}")

        if challenger_rmse < champion_rmse:
            self.champion_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(challenger, self.champion_model_path)
            print("Challenger promoted!")
            return True

        print("Champion retained. Challenger discarded.")
        return False

    @staticmethod
    def _ensure(value: object, message: str) -> None:
        if value is None:
            raise ValueError(message)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "data" / "processed" / "cleaned_pos_data.csv"
    champion_file = project_root / "models" / "xgboost_champion.pkl"

    retrainer = ChampionChallengerRetrainer(
        data_path=data_file,
        champion_model_path=champion_file,
        lag_steps=(7, 14),
        holdout_days=14,
    )

    result = retrainer.run()
    print("Retraining workflow finished.")
    print(f"Promoted: {result.promoted}")


if __name__ == "__main__":
    main()