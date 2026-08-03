"""Download raw kline data for the configured venue/symbol. Appends to an existing file if
present, otherwise downloads full history from 2017-01-01.
"""

from __future__ import annotations

from datetime import datetime

import click

from common.config import load_config, require_fields
from common.types import Venue
from inputs import get_download_functions


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "freq", "data_folder", "data_sources"])

    venue = Venue(config.get("venue", "binance"))
    data_sources = config["data_sources"]

    now = datetime.now()
    download_klines_fn = get_download_functions(venue)
    download_klines_fn(config, data_sources)

    elapsed = datetime.now() - now
    print(f"\nFinished downloading {len(data_sources)} data source(s) from {venue.value} in {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
