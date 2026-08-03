"""Shared CSV/parquet read-write and tail-windowing helpers used by every pipeline script.

Upstream ITB repeats this ~15-line block verbatim in every one of ``download.py`` through
``simulate.py``. Factoring it out here is a straightforward de-duplication, not a behavior
change — every script still windows to the same ``train_length``/``predict_length +
features_horizon`` size depending on ``config["train"]``, exactly as upstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_data_file(path: Path, time_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Data file does not exist: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=[time_column], date_format="ISO8601")
    raise ValueError(f"Unsupported file extension {path.suffix!r} (only .csv and .parquet are supported).")


def write_data_file(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False, float_format="%.6f")
    else:
        raise ValueError(f"Unsupported file extension {path.suffix!r} (only .csv and .parquet are supported).")


def window_size(config: dict) -> int | None:
    """Rows to keep, per ``config['train']`` mode, plus the online feature lookback horizon."""
    size = config.get("train_length") if config.get("train") else config.get("predict_length")
    features_horizon = config.get("features_horizon")
    if size and features_horizon:
        size += features_horizon
    return size or None


def tail_window(df: pd.DataFrame, size: int | None) -> pd.DataFrame:
    if size:
        df = df.tail(size).reset_index(drop=True)
    return df


def symbol_data_path(config: dict) -> Path:
    return Path(config["data_folder"]) / config["symbol"]
