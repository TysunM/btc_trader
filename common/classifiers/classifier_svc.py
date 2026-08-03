"""SVM classifier (SVC) / regressor (SVR) — exact port of upstream ITB's
``common/classifier_svc.py``. Same ``train_X``/``predict_X`` shape as the other classifier
modules (see ``classifier_lc.py``).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def train_svc(df_X: pd.DataFrame, df_y: pd.Series, model_config: dict) -> tuple:
    params = model_config.get("params", {})
    is_scale = params.get("is_scale", True)
    is_regression = params.get("is_regression", False)

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
    if is_regression:
        model = SVR(**args)
    else:
        args["probability"] = True  # required for predict_proba()
        model = SVC(**args)

    model.fit(X_train, y_train)
    return model, scaler


def predict_svc(model_pair: tuple, df_X_test: pd.DataFrame, model_config: dict) -> pd.Series:
    is_regression = model_config.get("params", {}).get("is_regression", False)
    model, scaler = model_pair
    input_index = df_X_test.index

    if scaler is not None:
        df_X_test = pd.DataFrame(data=scaler.transform(df_X_test), index=input_index)

    df_X_test_nonan = df_X_test.dropna()
    if is_regression:
        y_hat_nonan = model.predict(df_X_test_nonan.values)
    else:
        y_hat_nonan = model.predict_proba(df_X_test_nonan.values)[:, 1]
    y_hat_nonan = pd.Series(data=y_hat_nonan, index=df_X_test_nonan.index)

    out = pd.Series(index=input_index, dtype=float)
    out.loc[y_hat_nonan.index] = y_hat_nonan
    return out
