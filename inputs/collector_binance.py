"""Binance spot kline collector — bulk/offline download (used by ``scripts/download.py``, Phase 1)
and online incremental fetch + health check (used by the live service, Phase 3).

Ported from upstream ITB's ``inputs/collector_binance.py``. ``download_klines`` is fully
self-contained (a local ``Client``, no module state) and is all Phase 1 needs. ``init_client``/
``fetch_klines``/``health_check``/``close_client`` use a module-level ``client`` — this mirrors
upstream and is fine for a single-process, single-exchange-connection service, but Phase 3's
``service/app_state.py`` should hold the client reference explicitly rather than relying on this
module global, since that's exactly the kind of implicit shared state the project's instance-
based state deviation is meant to avoid for anything beyond a single quick script run.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from binance import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from common.utils import get_interval_count_from_start_dt, now_timestamp, pandas_get_interval, pandas_interval_length_ms
from inputs.utils_binance import binance_freq_from_pandas

log = logging.getLogger("binance_collector")

client: Client | None = None
append_overlap_records = 5

COLUMN_NAMES = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore",
]
COLUMN_TYPES = {
    "timestamp": "datetime64[ns, UTC]",
    "open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64",
    "close_time": "int64",
    "quote_av": "float64", "trades": "int64", "tb_base_av": "float64", "tb_quote_av": "float64",
    "ignore": "float64",
}
TIME_COLUMN = "timestamp"


def klines_to_df(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=COLUMN_NAMES)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], unit="ms", utc=True)
    df = df.astype(COLUMN_TYPES)
    df.set_index(TIME_COLUMN, inplace=True, drop=False)

    if df.isnull().any().any():
        null_columns = {k: v for k, v in df.isnull().any().to_dict().items() if v}
        log.warning(f"Null values in raw kline data. Columns with nulls: {null_columns}")

    return df


def download_klines(config: dict, data_sources: list[dict]) -> None:
    """Bulk-download (or incrementally extend) historic klines for each configured symbol."""
    time_column = config["time_column"]
    data_path = Path(config["data_folder"])
    download_max_rows = config.get("download_max_rows", 0)

    freq = config["freq"]
    binance_freq = binance_freq_from_pandas(freq)
    print(f"Pandas frequency: {freq}. Binance frequency: {binance_freq}")

    client_args = dict(config.get("client_args", {}))
    if config.get("api_key"):
        client_args["api_key"] = config["api_key"]
    if config.get("api_secret"):
        client_args["api_secret"] = config["api_secret"]

    local_client = Client(**client_args)

    for ds in data_sources:
        quote = ds.get("folder")
        if not quote:
            print("ERROR: data source 'folder' is not specified.")
            continue

        print(f"Start downloading {quote!r}...")

        file_path = data_path / quote
        file_path.mkdir(parents=True, exist_ok=True)
        file_name = (file_path / ds.get("file", "klines")).with_suffix(".csv")

        if file_name.is_file():
            df = pd.read_csv(file_name)
            df[time_column] = pd.to_datetime(df[time_column], format="ISO8601", utc=True)
            df = df.astype(COLUMN_TYPES)
            df = df.set_index(time_column, drop=False)
            # Use an older point so newly downloaded data overwrites the last few (possibly
            # revised) rows rather than assuming the file's tail was already final.
            oldest_point = df["timestamp"].iloc[-5] if len(df) >= 5 else df["timestamp"].iloc[0]
            print(f"File found. Appending {quote!r}/{freq} data since {oldest_point} to {file_name}")
        else:
            df = None
            oldest_point = datetime(2017, 1, 1)
            print(f"File not found. Downloading all available history for {quote!r}/{freq} to {file_name}")

        klines = local_client.get_historical_klines(
            symbol=quote,
            interval=binance_freq,
            start_str=oldest_point.isoformat(),
        )
        df_new = klines_to_df(klines)

        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new])
            df = df.drop_duplicates(subset=["timestamp"], keep="last")

        # Drop the last row: it's the still-open (incomplete) current candle.
        df = df.iloc[:-1]

        if download_max_rows:
            df = df.tail(download_max_rows)

        df.to_csv(file_name, index=False)
        print(f"Finished downloading {quote!r}. Stored {len(df)} rows in {file_name}")


def init_client(parameters: dict, client_args: dict) -> None:
    global client, append_overlap_records
    append_overlap_records = parameters.get("append_overlap_records", 5)
    client = Client(**client_args)


def get_client() -> Client:
    return client


def close_client() -> None:
    if client is not None:
        client.close_connection()


async def request_symbol_klines(symbol: str, freq: str, limit: int) -> dict[str, list]:
    """Request up to ``limit`` recent klines for one symbol, excluding the still-open candle."""
    klines_per_request = 400

    now_ts = now_timestamp()
    start_ts, _ = pandas_get_interval(freq)
    binance_freq = binance_freq_from_pandas(freq)
    interval_length_ms = pandas_interval_length_ms(freq)

    try:
        if limit <= klines_per_request:
            klines = client.get_klines(symbol=symbol, interval=binance_freq, limit=limit, endTime=now_ts)
        else:
            request_start_ts = now_ts - interval_length_ms * (limit + 1)
            klines = client.get_historical_klines(
                symbol=symbol, interval=binance_freq, start_str=request_start_ts, end_str=now_ts
            )
    except (BinanceRequestException, BinanceAPIException) as e:
        log.error(f"Binance error while requesting klines for {symbol}: {e}")
        return {}
    except Exception as e:
        log.error(f"Exception while requesting klines for {symbol}: {e}")
        return {}

    klines_full = [kl for kl in klines if kl[0] < start_ts]
    if not klines_full:
        return {symbol: []}

    last_full_kline_ts = klines_full[-1][0]
    if last_full_kline_ts != start_ts - interval_length_ms:
        log.error(
            f"Unexpected result for {symbol}: last full kline ts {last_full_kline_ts} != "
            f"expected {start_ts - interval_length_ms}. Possible gap in data."
        )

    return {symbol: klines_full}


async def fetch_klines(config: dict, start_from_dt) -> dict[str, pd.DataFrame] | None:
    """Fetch all klines needed to bring the local window up to date, starting from ``start_from_dt``."""
    data_sources = config.get("data_sources", [])
    symbols = [x.get("folder") for x in data_sources] or [config["symbol"]]
    freq = config["freq"]

    intervals_count = get_interval_count_from_start_dt(freq, start_from_dt)
    request_count = intervals_count + append_overlap_records

    tasks = [asyncio.create_task(request_symbol_klines(sym, freq, request_count)) for sym in symbols]

    results: dict[str, list] = {}
    timeout = 10
    try:
        for fut in asyncio.as_completed(tasks, timeout=timeout):
            res = await fut
            if res:
                results.update(res)
            else:
                log.error("Received empty result from a klines request.")
                return None
    except TimeoutError:
        log.warning(f"Timeout ({timeout}s) while requesting kline data.")
        return None

    return {symbol: klines_to_df(klines) for symbol, klines in results.items()}


async def health_check() -> int:
    """Return 0 if the Binance system status is normal, 1 otherwise."""
    system_status = client.get_system_status()
    if not system_status or system_status.get("status") != 0:
        log.error(f"Binance system status is not normal: {system_status}")
        return 1
    return 0
