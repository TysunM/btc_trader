"""Walk-forward backtest: repeatedly train on a preceding window and predict the next window,
so every prediction is made on data the model never saw during training. This is what makes
the resulting trade-performance numbers (fed into signals.py / simulate.py) trustworthy.

v2 addition (CHANGELOG_V2.md item 2): an optional ``rolling_predict.embargo`` config value
extends the training-cutoff gap before each predict window beyond upstream's existing (already
sufficient) minimum, as an extra conservatism margin. Defaults to 0 rows (v1 baseline, unchanged
behavior) — see ``v2/purge_embargo.py`` for why this is an optional safety margin rather than a
bug fix.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

import click
from joblib import Parallel, delayed

from common.config import load_config, require_fields
from common.generators import predict_feature_set, train_feature_set
from common.io import read_data_file, symbol_data_path, write_data_file
from common.model_store import ModelStore, score_to_label_algo_pair
from common.utils import compute_scores, compute_scores_regression, find_index
from v2.purge_embargo import purged_train_end


def execute_train_predict_step(
    config: dict, model_store: ModelStore, train_df: pd.DataFrame, predict_df: pd.DataFrame, parallel
) -> pd.DataFrame:
    """Train fresh models on ``train_df`` (optionally in parallel across train_feature_sets
    entries), persist them, then predict on ``predict_df`` using the freshly trained models."""
    train_feature_sets = config.get("train_feature_sets", [])

    print(f"  Training {len(train_feature_sets)} feature set(s) on {len(train_df)} rows...")
    models: dict[str, tuple] = {}
    calibrators: dict[str, object] = {}
    if isinstance(parallel, Parallel):
        for fs_models, fs_calibrators in parallel(delayed(train_feature_set)(train_df, fs, config) for fs in train_feature_sets):
            models.update(fs_models)
            calibrators.update(fs_calibrators)
    else:
        for fs in train_feature_sets:
            fs_models, fs_calibrators = train_feature_set(train_df, fs, config)
            models.update(fs_models)
            calibrators.update(fs_calibrators)

    # predict_feature_set expects models to be retrievable from the model store, and the
    # freshly-trained models here supersede whatever was loaded from disk for this step.
    for score_column_name, model_pair in models.items():
        model_store.put_model_pair(score_column_name, model_pair)
    for score_column_name, calibrator in calibrators.items():
        model_store.put_calibrator(score_column_name, calibrator)

    print(f"  Predicting {len(predict_df)} rows...")
    out_df = pd.DataFrame()
    for fs in train_feature_sets:
        fs_out_df, _ = predict_feature_set(predict_df, fs, config, model_store)
        out_df = pd.concat([out_df, fs_out_df], axis=1)

    return out_df


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(
        config,
        ["symbol", "data_folder", "time_column", "train_features", "labels", "train_feature_sets", "rolling_predict"],
    )

    model_store = ModelStore(config)

    time_column = config["time_column"]
    data_path = symbol_data_path(config)
    now = datetime.now()

    file_path = data_path / config["matrix_file_name"]
    print(f"Loading data from {file_path}...")
    df = read_data_file(file_path, time_column)
    print(f"Loaded {len(df)} records / {len(df.columns)} columns.")

    rp_config = config["rolling_predict"]
    data_start = rp_config.get("data_start")
    data_end = rp_config.get("data_end")

    if data_start:
        df = df[df[time_column] >= data_start] if isinstance(data_start, str) else df.iloc[data_start:]
    if data_end:
        df = df[df[time_column] < data_end] if isinstance(data_end, str) else df.iloc[:-data_end]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    prediction_start = rp_config.get("prediction_start")
    if isinstance(prediction_start, str):
        prediction_start = find_index(df, prediction_start, time_column)
    prediction_size = rp_config.get("prediction_size")
    prediction_steps = rp_config.get("prediction_steps")

    if not prediction_start:
        if not prediction_size or not prediction_steps:
            raise ValueError("Only one of prediction_start/prediction_size/prediction_steps may be empty.")
        prediction_start = len(df) - prediction_size * prediction_steps
    elif not prediction_size:
        prediction_size = (len(df) - prediction_start) // prediction_steps
    elif not prediction_steps:
        prediction_steps = (len(df) - prediction_start) // prediction_size

    if len(df) - prediction_start < prediction_steps * prediction_size:
        raise ValueError(
            f"Not enough data for {prediction_steps} steps of size {prediction_size} starting at {prediction_start}: "
            f"only {len(df) - prediction_start} rows available."
        )

    train_features_all = config["train_features"]
    labels_all = config["labels"]

    out_columns = [c for c in [time_column, "open", "high", "low", "close", "volume", "close_time"] if c in df.columns]
    labels_present = set(labels_all).issubset(df.columns)
    all_features = train_features_all + labels_all if labels_present else train_features_all
    df = df[out_columns + [x for x in all_features if x not in out_columns]]

    for label in labels_all:
        if np.issubdtype(df[label].dtype, np.bool_):
            df[label] = df[label].astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    print(f"Start index: {prediction_start}. Steps: {prediction_steps}. Step size: {prediction_size}")

    label_horizon = config["label_horizon"]
    train_length = config.get("train_length")
    embargo = rp_config.get("embargo", 0)  # v2 item 2 (CHANGELOG_V2.md): 0 = v1 baseline

    use_multiprocessing = rp_config.get("use_multiprocessing", False)
    max_workers = rp_config.get("max_workers")
    parallel = Parallel(n_jobs=max_workers, backend="loky") if use_multiprocessing else None

    labels_hat_df = pd.DataFrame()
    for step in range(prediction_steps):
        predict_start = prediction_start + step * prediction_size
        predict_end = predict_start + prediction_size
        predict_df = df.iloc[predict_start:predict_end]

        # Exclude rows near the prediction window whose labels look forward into it —
        # otherwise the "future" the model is being tested on has already leaked into training.
        train_end = purged_train_end(predict_start, label_horizon, embargo)
        train_start = max(0, train_end - train_length) if train_length else 0
        train_df = df.iloc[train_start:train_end].dropna(subset=train_features_all)

        print(
            f"\n=== Step {step}/{prediction_steps}. Train range [{train_start}, {train_end}] "
            f"({train_end - train_start} rows). Predict range [{predict_start}, {predict_end}] "
            f"({predict_end - predict_start} rows)"
        )
        step_start = datetime.now()
        predict_labels_df = execute_train_predict_step(config, model_store, train_df, predict_df, parallel)
        labels_hat_df = pd.concat([labels_hat_df, predict_labels_df])
        print(f"  Step done in {str(datetime.now() - step_start).split('.')[0]}")

    print(f"\nFinished {prediction_steps} steps, {len(labels_hat_df)} predicted rows.")

    out_df = labels_hat_df.join(df[out_columns + labels_all])
    out_path = data_path / config["predict_file_name"]
    print(f"Storing {len(out_df)} records / {len(out_df.columns)} columns to {out_path}...")
    write_data_file(out_df, out_path)

    score_lines = []
    for score_column_name in labels_hat_df.columns:
        label_column, _ = score_to_label_algo_pair(score_column_name)
        df_scores = pd.DataFrame({"y_true": out_df[label_column], "y_predicted": out_df[score_column_name]}).dropna()
        y_true, y_predicted = df_scores["y_true"], df_scores["y_predicted"]
        if ptypes.is_float_dtype(y_true) and ptypes.is_float_dtype(y_predicted):
            score = compute_scores_regression(y_true, y_predicted)
        else:
            score = compute_scores(y_true.astype(int), y_predicted)
        score_lines.append(f"{score_column_name}: {score}")

    score_path = out_path.with_suffix(".txt")
    with open(score_path, "a+") as f:
        f.write("\n".join(score_lines) + "\n\n")
    print(f"Prediction scores stored in: {score_path}")

    print(f"Finished rolling prediction in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
