"""Merge multiple raw data-source files into one continuous time-raster table (``data.csv``)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from common.config import load_config, require_fields
from common.io import read_data_file, symbol_data_path, tail_window, window_size, write_data_file
from common.utils import merge_data_sources


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "freq", "data_folder", "data_sources", "time_column"])

    time_column = config["time_column"]
    data_path = Path(config["data_folder"])
    ws = window_size(config)

    now = datetime.now()

    data_sources = config.get("data_sources", [])
    if not data_sources:
        print("ERROR: data_sources is empty. Nothing to merge.")
        return

    for ds in data_sources:
        quote = ds.get("folder")
        if not quote:
            print("ERROR: data source 'folder' is not specified.")
            continue
        file_name = ds.get("file", quote)
        file_path = (data_path / quote / file_name)
        if not file_path.suffix:
            file_path = file_path.with_suffix(".csv")

        print(f"Reading data file: {file_path}")
        df = read_data_file(file_path, time_column)
        print(f"Loaded {len(df)} records.")
        ds["df"] = tail_window(df, ws)

    freq = config["freq"]
    merge_interpolate = config.get("merge_interpolate", False)
    df_out = merge_data_sources(data_sources, time_column, freq, merge_interpolate)

    out_path = symbol_data_path(config) / config["merge_file_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Storing merged output file...")
    df_out = df_out.reset_index(drop=(df_out.index.name in df_out.columns))
    write_data_file(df_out, out_path)

    print(f"Stored {out_path} with {len(df_out)} records. Range: ({df_out[time_column].iloc[0]}, {df_out[time_column].iloc[-1]})")
    print(f"Finished merging data in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
