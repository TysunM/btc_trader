"""Keras MLP classifier/regressor — ported from upstream ITB's ``common/classifier_nn.py``.

Requires the optional ``nn`` extra (``uv sync --extra nn``) — TensorFlow/Keras are the heaviest,
most version-sensitive dependency in this stack, so nothing else in the pipeline imports this
module unless a config actually declares an ``algo: "nn"`` entry (see the lazy imports in
``common/generators.py``).
"""

from __future__ import annotations

import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


def train_nn(df_X: pd.DataFrame, df_y: pd.Series, model_config: dict) -> tuple:
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

    n_features = X_train.shape[1]
    layers = params.get("layers") or [n_features // 4]
    if not isinstance(layers, list):
        layers = [layers]

    model = Sequential()
    for i, out_features in enumerate(layers):
        in_features = n_features if i == 0 else layers[i - 1]
        model.add(Dense(out_features, activation="sigmoid", input_dim=in_features))

    train_conf = model_config.get("train", {})
    learning_rate = train_conf.get("learning_rate")
    n_epochs = train_conf.get("n_epochs")
    batch_size = train_conf.get("bs")

    if is_regression:
        model.add(Dense(units=1))
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error"),
                tf.keras.metrics.R2Score(name="r2_score"),
            ],
        )
    else:
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    es_args = dict(monitor="loss", min_delta=0.00001, patience=5, verbose=0, mode="auto")
    es_args.update(train_conf.get("es", {}))
    early_stopping = EarlyStopping(**es_args)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, callbacks=[early_stopping], verbose=1)

    return model, scaler


def predict_nn(model_pair: tuple, df_X_test: pd.DataFrame, model_config: dict) -> pd.Series:
    model, scaler = model_pair
    input_index = df_X_test.index

    if scaler is not None:
        df_X_test = pd.DataFrame(data=scaler.transform(df_X_test), index=input_index)

    df_X_test_nonan = df_X_test.dropna()

    # Reset Keras's global state — otherwise repeated predict calls (e.g. across
    # predict_rolling.py's walk-forward steps) leak memory.
    tf.keras.backend.clear_session()

    y_hat_nonan = model.predict_on_batch(df_X_test_nonan.values)[:, 0]
    y_hat_nonan = pd.Series(data=y_hat_nonan, index=df_X_test_nonan.index)

    out = pd.Series(index=input_index, dtype=float)
    out.loc[y_hat_nonan.index] = y_hat_nonan
    return out
