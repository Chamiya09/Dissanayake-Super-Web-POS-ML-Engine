from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from member4_data_preparation import prepare_model_datasets, time_based_train_test_split
from member6_evaluation import metric_pack


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
    train_df, test_df = time_based_train_test_split(df, date_col)
    X_train, X_test, y_train, y_test = prepare_model_datasets(train_df, test_df, feature_cols)

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
