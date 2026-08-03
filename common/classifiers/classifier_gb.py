"""LightGBM classifier/regressor — exact port of upstream ITB's ``common/classifier_gb.py``.
Same ``train_X``/``predict_X`` shape as the other classifier modules (see ``classifier_lc.py``).
"""

from __future__ import annotations

import lightgbm as lgbm
import pandas as pd
from sklearn.preprocessing import StandardScaler


def train_gb(df_X: pd.DataFrame, df_y: pd.Series, model_config: dict) -> tuple:
    params = model_config.get("params", {})
    is_scale = params.get("is_scale", False)

    if is_scale:
        scaler = StandardScaler()
        scaler.fit(df_X)
        X_train = scaler.transform(df_X)
    else:
        scaler = None
        X_train = df_X.values

    y_train = df_y.values

    train_conf = model_config.get("train", {})
    # See https://lightgbm.readthedocs.io/en/latest/Parameters.html
    model = lgbm.train(dict(train_conf), train_set=lgbm.Dataset(X_train, y_train))

    return model, scaler


def predict_gb(model_pair: tuple, df_X_test: pd.DataFrame, model_config: dict) -> pd.Series:
    model, scaler = model_pair
    input_index = df_X_test.index

    if scaler is not None:
        df_X_test = pd.DataFrame(data=scaler.transform(df_X_test), index=input_index)

    df_X_test_nonan = df_X_test.dropna()
    y_hat_nonan = model.predict(df_X_test_nonan.values)
    y_hat_nonan = pd.Series(data=y_hat_nonan, index=df_X_test_nonan.index)

    out = pd.Series(index=input_index, dtype=float)
    out.loc[y_hat_nonan.index] = y_hat_nonan
    return out
