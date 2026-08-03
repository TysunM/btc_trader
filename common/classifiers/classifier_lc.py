"""Logistic-regression classifier — exact port of upstream ITB's ``common/classifier_lc.py``.

Every classifier module in this package shares the same shape:
``train_X(df_X, df_y, model_config) -> (model, scaler)`` and
``predict_X(model_pair, df_X_test, model_config) -> pd.Series`` (NaN-safe: rows with NaN
features predict NaN rather than raising or silently dropping).
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_lc(df_X: pd.DataFrame, df_y: pd.Series, model_config: dict) -> tuple:
    params = model_config.get("params", {})
    is_scale = params.get("is_scale", True)

    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    train_conf = model_config.get("train", {})
    args = dict(train_conf)
    args["verbose"] = 0
    model = LogisticRegression(**args)
    model.fit(X_train, y_train)

    return model, scaler


def predict_lc(model_pair: tuple, df_X_test: pd.DataFrame, model_config: dict) -> pd.Series:
    model, scaler = model_pair
    input_index = df_X_test.index

    if scaler is not None:
        df_X_test = pd.DataFrame(data=scaler.transform(df_X_test), index=input_index)

    df_X_test_nonan = df_X_test.dropna()
    y_hat_nonan = model.predict_proba(df_X_test_nonan.values)[:, 1]
    y_hat_nonan = pd.Series(data=y_hat_nonan, index=df_X_test_nonan.index)

    out = pd.Series(index=input_index, dtype=float)
    out.loc[y_hat_nonan.index] = y_hat_nonan
    return out
