"""Apply all configured label generators to the feature table, producing the full
feature+label matrix used for training and prediction.
"""

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
    require_fields(config, ["symbol", "data_folder", "time_column", "label_sets"])

    model_store = ModelStore(config)
    model_store.load_models()

    time_column = config["time_column"]
    data_path = symbol_data_path(config)
    now = datetime.now()

    file_path = data_path / config["feature_file_name"]
    print(f"Loading data from {file_path}...")
    df = read_data_file(file_path, time_column)
    df = tail_window(df, window_size(config))
    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    label_sets = config.get("label_sets", [])
    if not label_sets:
        print("ERROR: no label_sets defined. Nothing to process.")
        return

    print(f"Start generating labels for {len(df)} input records.")
    all_labels: list[str] = []
    for i, fs in enumerate(label_sets):
        fs_now = datetime.now()
        print(f"Label set {i}/{len(label_sets)}: {fs.get('generator')}...")
        df, new_labels = generate_feature_set(df, fs, config, model_store, last_rows=0)
        all_labels.extend(new_labels)
        print(f"  -> {len(new_labels)} labels. {str(datetime.now() - fs_now).split('.')[0]}")

    df = df.replace([np.inf, -np.inf], np.nan)
    na_count = df[all_labels].isna().any(axis=1).sum()
    if na_count:
        print(f"WARNING: {na_count} rows have NaN in some label column(s) (expected at the tail — future data not yet available).")

    out_path = data_path / config["matrix_file_name"]
    print(f"Storing {len(df)} records / {len(df.columns)} columns to {out_path}...")
    write_data_file(df, out_path)

    with open(out_path.with_suffix(".txt"), "a+") as f:
        f.write(", ".join(f'"{x}"' for x in all_labels) + "\n\n")

    print(f"Finished generating {len(all_labels)} labels in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
