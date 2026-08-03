"""Execute output/notification generators against the signals file — mainly useful for
replaying a historic signals.csv through the notifier/trader code paths for testing, since the
live service (Phase 3) calls the same ``output_feature_set`` dispatcher on each tick.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import click

from common.config import load_config, require_fields
from common.generators import output_feature_set
from common.io import read_data_file, symbol_data_path
from common.model_store import ModelStore


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "data_folder", "time_column", "output_sets"])

    model_store = ModelStore(config)
    model_store.load_models()

    time_column = config["time_column"]
    data_path = symbol_data_path(config)
    now = datetime.now()

    file_path = data_path / config["signal_file_name"]
    print(f"Loading signals from {file_path}...")
    df = read_data_file(file_path, time_column)
    df = df.set_index(time_column, drop=False)
    print(f"Loaded {len(df)} records / {len(df.columns)} columns.")

    output_sets = config.get("output_sets", [])
    for output_set in output_sets:
        try:
            asyncio.run(output_feature_set(df, output_set, config, model_store))
        except Exception as e:
            print(f"ERROR in output generator {output_set.get('generator')!r}: {e}")
            return

    print(f"Finished executing outputs in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
