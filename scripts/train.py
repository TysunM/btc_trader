"""Train one model per (label x algorithm) combination declared in train_feature_sets."""

from __future__ import annotations

from datetime import datetime

import numpy as np

import click

from common.config import load_config, require_fields
from common.generators import train_feature_set
from common.io import read_data_file, symbol_data_path, tail_window, window_size, write_data_file
from common.model_store import ModelStore


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "data_folder", "time_column", "train_features", "labels", "train_feature_sets"])

    model_store = ModelStore(config)

    time_column = config["time_column"]
    data_path = symbol_data_path(config)
    now = datetime.now()

    if not config.get("train"):
        print("WARNING: config['train'] is false; training will proceed anyway with a short (predict_length) window.")

    file_path = data_path / config["matrix_file_name"]
    print(f"Loading data from {file_path}...")
    df = read_data_file(file_path, time_column)
    df = tail_window(df, window_size(config))  # broad window (train_length + features_horizon)
    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    train_features_all = config["train_features"]
    labels_all = config["labels"]

    out_columns = [c for c in [time_column, "open", "high", "low", "close", "volume", "close_time"] if c in df.columns]
    all_features = train_features_all + labels_all
    df = df[out_columns + [x for x in all_features if x not in out_columns]]

    for label in labels_all:
        if np.issubdtype(df[label].dtype, np.bool_):
            df[label] = df[label].astype(int)

    label_horizon = config["label_horizon"]
    train_length = config.get("train_length")

    if label_horizon:
        df = df.head(-label_horizon)
    if train_length:
        df = df.tail(train_length)

    df = df.replace([np.inf, -np.inf], np.nan)
    na_count = df[train_features_all].isna().any(axis=1).sum()
    if na_count:
        print(f"WARNING: {na_count} rows have NaN in some feature column(s).")
    df = df.reset_index(drop=True)

    train_feature_sets = config.get("train_feature_sets", [])
    if not train_feature_sets:
        print("ERROR: no train_feature_sets defined. Nothing to process.")
        return

    print(f"Start training models on {len(df)} input records.")
    models: dict[str, tuple] = {}
    calibrators: dict[str, object] = {}
    for i, fs in enumerate(train_feature_sets):
        fs_now = datetime.now()
        print(f"Train feature set {i}/{len(train_feature_sets)}: {fs.get('generator')}...")
        fs_models, fs_calibrators = train_feature_set(df, fs, config)
        models.update(fs_models)
        calibrators.update(fs_calibrators)
        print(f"  -> {str(datetime.now() - fs_now).split('.')[0]}")

    for score_column_name, model_pair in models.items():
        model_store.put_model_pair(score_column_name, model_pair)
    for score_column_name, calibrator in calibrators.items():
        model_store.put_calibrator(score_column_name, calibrator)

    print(f"Models stored in: {model_store.model_path}" + (f" ({len(calibrators)} calibrated)" if calibrators else ""))
    print(f"Finished training in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
