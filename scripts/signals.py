"""Generate trade signal columns from the rolling prediction scores (combine + threshold rules)."""

from __future__ import annotations

from datetime import datetime

import numpy as np

import click

from common.config import load_config, require_fields
from common.generators import generate_feature_set
from common.io import read_data_file, symbol_data_path, tail_window, window_size, write_data_file
from common.model_store import ModelStore


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "data_folder", "time_column", "signal_sets"])

    model_store = ModelStore(config)
    model_store.load_models()

    time_column = config["time_column"]
    data_path = symbol_data_path(config)
    now = datetime.now()

    file_path = data_path / config["predict_file_name"]
    print(f"Loading data from {file_path}...")
    df = read_data_file(file_path, time_column)
    df = tail_window(df, window_size(config))
    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    signal_sets = config.get("signal_sets", [])
    if not signal_sets:
        print("ERROR: no signal_sets defined. Nothing to process.")
        return

    print(f"Start generating signals for {len(df)} input records.")
    all_features: list[str] = []
    for i, fs in enumerate(signal_sets):
        fs_now = datetime.now()
        print(f"Signal set {i}/{len(signal_sets)}: {fs.get('generator')}...")
        df, new_features = generate_feature_set(df, fs, config, model_store, last_rows=0)
        all_features.extend(new_features)
        print(f"  -> {len(new_features)} column(s). {str(datetime.now() - fs_now).split('.')[0]}")

    df = df.replace([np.inf, -np.inf], np.nan)
    na_count = df[all_features].isna().any(axis=1).sum()
    if na_count:
        print(f"WARNING: {na_count} rows have NaN in some signal column(s).")

    out_columns = [c for c in [time_column, "open", "high", "low", "close"] if c in df.columns]
    out_columns += [c for c in config.get("labels", []) if c in df.columns]
    out_columns += all_features
    out_df = df[out_columns]

    out_path = data_path / config["signal_file_name"]
    print(f"Storing {len(out_df)} records / {len(out_df.columns)} columns to {out_path}...")
    write_data_file(out_df, out_path)

    print(f"Finished signal generation in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
