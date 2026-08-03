"""Apply trained models to the feature matrix and compute prediction scores."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

import click

from common.config import load_config, require_fields
from common.generators import predict_feature_set
from common.io import read_data_file, symbol_data_path, tail_window, window_size, write_data_file
from common.model_store import ModelStore, score_to_label_algo_pair
from common.utils import compute_scores, compute_scores_regression


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "data_folder", "time_column", "train_features", "labels", "train_feature_sets"])

    model_store = ModelStore(config)
    model_store.load_models()

    time_column = config["time_column"]
    data_path = symbol_data_path(config)
    now = datetime.now()

    if config.get("train"):
        print("WARNING: config['train'] is true; this script only predicts, it does not train.")

    file_path = data_path / config["matrix_file_name"]
    print(f"Loading data from {file_path}...")
    df = read_data_file(file_path, time_column)
    df = tail_window(df, window_size(config))
    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    train_features_all = config["train_features"]
    labels_all = config["labels"]

    out_columns = [c for c in [time_column, "open", "high", "low", "close", "volume", "close_time"] if c in df.columns]
    labels_present = set(labels_all).issubset(df.columns)
    all_features = train_features_all + labels_all if labels_present else train_features_all
    df = df[out_columns + [x for x in all_features if x not in out_columns]]

    df = df.replace([np.inf, -np.inf], np.nan)
    na_count = df[train_features_all].isna().any(axis=1).sum()
    if na_count:
        print(f"WARNING: {na_count} rows have NaN feature(s); dropping them.")
        df = df.dropna(subset=train_features_all).reset_index(drop=True)

    train_feature_sets = config.get("train_feature_sets", [])
    if not train_feature_sets:
        print("ERROR: no train_feature_sets defined. Nothing to process.")
        return

    print(f"Start predicting for {len(df)} input records.")
    labels_hat_df = pd.DataFrame()
    for i, fs in enumerate(train_feature_sets):
        fs_now = datetime.now()
        print(f"Train feature set {i}/{len(train_feature_sets)}: {fs.get('generator')}...")
        fs_out_df, _ = predict_feature_set(df, fs, config, model_store)
        labels_hat_df = pd.concat([labels_hat_df, fs_out_df], axis=1)
        print(f"  -> {str(datetime.now() - fs_now).split('.')[0]}")

    out_df = labels_hat_df.join(df[out_columns + (labels_all if labels_present else [])])

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

    print(f"Finished predicting in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
