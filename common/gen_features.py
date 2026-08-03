"""Feature generators. A feature generator knows how to compute derived features from its
declarative specification in the config file.

``generate_features_talib`` — the generator ITB's own sample configs use for every feature — is
backed by :mod:`common.ta_adapter` instead of real ``talib`` (see that module's docstring for
why). ``generate_features_itblib`` and ``generate_features_itbstats`` (Phase 2) are ported
faithfully from upstream, using the rolling-aggregation helpers in
:mod:`common.gen_features_rolling_agg`. ``tsfresh`` (optional upstream dependency) and ``depth``
(order-book depth data, irrelevant to a BTCUSDT spot OHLCV pipeline) remain unported — see
``common/generators.py``'s dispatch table.

Simplification vs. upstream: the original ``generate_features_talib`` also supports relative/
percentage/log post-transforms (``rel_base``, ``rel_func``, ``percentage``, ``log``) and a talib
"stream" fast-path for online incremental computation. None of ITB's own sample configs actually
set those transform parameters, so this port implements the ``{columns, functions, windows}``
shape they do use and leaves the transform options as a documented gap rather than unused
complexity — revisit if a config that needs them shows up.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import ta_adapter
from common.gen_features_rolling_agg import (
    add_area_ratio,
    add_future_aggregations,
    add_linear_trends,
    add_past_aggregations,
    add_past_weighted_aggregations,
    to_diff,
)


def generate_features_talib(df: pd.DataFrame, config: dict, last_rows: int = 0) -> list[str]:
    """config: {"columns": [...] | str, "functions": [...] | str, "windows": [...] | int}

    Mutates ``df`` in place (adds one column per column x function x window combination) and
    returns the list of generated column names — this in-place-mutate-and-return-names-only
    shape matches upstream exactly, since ``common/generators.py``'s dispatcher relies on it.
    """
    columns = config.get("columns")
    if isinstance(columns, str):
        columns = [columns]

    func_names = config.get("functions")
    if not isinstance(func_names, list):
        func_names = [func_names]

    windows = config.get("windows")
    if not isinstance(windows, list):
        windows = [windows]

    features: list[str] = []
    for col in columns:
        series = df[col].interpolate()  # talib-style: one stray NaN can poison a whole window
        for func_name in func_names:
            for window in windows:
                out_name = f"{col}_{func_name}_{window}"
                df[out_name] = ta_adapter.call(func_name, series, window)
                features.append(out_name)

    return features


def generate_features_itblib(df: pd.DataFrame, config: dict, last_rows: int = 0) -> list[str]:
    """ITB's original hand-rolled feature set: rolling means/stds of close/volume/span/trades/
    taker-buy-ratio (relative to a longer base window), plus area-ratio and linear-trend
    features. Column names look like ``close_120`` (average close over the last 120 bars,
    relative to the base window).
    """
    use_differences = config.get("use_differences", True)
    base_window = config.get("base_window", True)
    windows = config.get("windows", True)
    functions = config.get("functions", True)

    features: list[str] = []
    to_drop: list[str] = []

    if use_differences:
        df["close"] = to_diff(df["close"])
        df["volume"] = to_diff(df["volume"])
        df["trades"] = to_diff(df["trades"])

    if not functions or "close_WMA" in functions:
        to_drop += add_past_weighted_aggregations(df, "close", "volume", np.nanmean, base_window, suffix="", last_rows=last_rows)
        features += add_past_weighted_aggregations(df, "close", "volume", np.nanmean, windows, "", to_drop[-1], 100.0, last_rows=last_rows)

    if not functions or "close_STD" in functions:
        to_drop += add_past_aggregations(df, "close", np.nanstd, base_window, last_rows=last_rows)
        features += add_past_aggregations(df, "close", np.nanstd, windows, "_std", to_drop[-1], 100.0, last_rows=last_rows)

    if not functions or "volume_SMA" in functions:
        to_drop += add_past_aggregations(df, "volume", np.nanmean, base_window, suffix="", last_rows=last_rows)
        features += add_past_aggregations(df, "volume", np.nanmean, windows, "", to_drop[-1], 100.0, last_rows=last_rows)

    if not functions or "span_SMA" in functions:
        df["span"] = df["high"] - df["low"]
        to_drop.append("span")
        to_drop += add_past_aggregations(df, "span", np.nanmean, base_window, suffix="", last_rows=last_rows)
        features += add_past_aggregations(df, "span", np.nanmean, windows, "", to_drop[-1], 100.0, last_rows=last_rows)

    if not functions or "trades_SMA" in functions:
        to_drop += add_past_aggregations(df, "trades", np.nanmean, base_window, suffix="", last_rows=last_rows)
        features += add_past_aggregations(df, "trades", np.nanmean, windows, "", to_drop[-1], 100.0, last_rows=last_rows)

    if not functions or "tb_base_SMA" in functions:
        df["tb_base"] = df["tb_base_av"] / df["volume"]
        to_drop.append("tb_base")
        to_drop += add_past_aggregations(df, "tb_base", np.nanmean, base_window, suffix="", last_rows=last_rows)
        features += add_past_aggregations(df, "tb_base", np.nanmean, windows, "", to_drop[-1], 100.0, last_rows=last_rows)

    if not functions or "close_AREA" in functions:
        features += add_area_ratio(df, is_future=False, column_name="close", windows=windows, suffix="_area", last_rows=last_rows)

    if not functions or "close_SLOPE" in functions:
        features += add_linear_trends(df, is_future=False, column_name="close", windows=windows, suffix="_trend", last_rows=last_rows)
    if not functions or "volume_SLOPE" in functions:
        features += add_linear_trends(df, is_future=False, column_name="volume", windows=windows, suffix="_trend", last_rows=last_rows)

    df.drop(columns=to_drop, inplace=True)
    return features


def fmax_fn(x: np.ndarray) -> float:
    return np.argmax(x) / len(x) if len(x) > 0 else np.nan


def lsbm_fn(x: np.ndarray) -> float:
    """Longest consecutive run of values below the window mean (tsfresh's
    ``longest_strike_below_mean``, reimplemented to avoid the tsfresh dependency)."""

    def _run_lengths_where(mask: np.ndarray) -> list[int]:
        if len(mask) == 0:
            return [0]
        res = [len(list(g)) for v, g in itertools.groupby(mask) if v == 1]
        return res or [0]

    return np.max(_run_lengths_where(x < np.mean(x))) if x.size > 0 else 0


def generate_features_itbstats(df: pd.DataFrame, config: dict, last_rows: int = 0) -> list[str]:
    """Statistical features (skew, kurtosis, longest-strike-below-mean, first-location-of-max,
    mean, std, area, slope) over rolling windows of one column.
    """
    column_names = config.get("columns")
    if isinstance(column_names, str):
        column_name = column_names
    elif isinstance(column_names, list):
        column_name = column_names[0]
    elif isinstance(column_names, dict):
        column_name = next(iter(column_names.values()))
    else:
        raise ValueError(f"'columns' must be a string, list, or dict, got {type(column_names)}")

    column = df[column_name].interpolate()

    func_names = config.get("functions")
    if not isinstance(func_names, list):
        func_names = [func_names]

    windows = config.get("windows")
    if not isinstance(windows, list):
        windows = [windows]

    bias = config.get("parameters", {}).get("bias", False)

    features: list[str] = []
    for func_name in func_names:
        name = func_name.lower()
        args: tuple = ()
        if name == "scipy_skew":
            fn, args = stats.skew, (0, bias)
        elif name == "pandas_skew":
            fn = lambda x: pd.Series(x).skew()
        elif name == "scipy_kurtosis":
            fn, args = stats.kurtosis, (0, bias)
        elif name == "pandas_kurtosis":
            fn = lambda x: pd.Series(x).kurtosis()
        elif name == "lsbm":
            fn = lsbm_fn
        elif name == "fmax":
            fn = fmax_fn
        elif name == "mean":
            fn = np.nanmean
        elif name == "std":
            fn = np.nanstd
        else:
            raise ValueError(f"Unknown itbstats function {func_name!r}.")

        for w in windows:
            out_name = f"{column_name}_{func_name}_{w}"
            ro = column.rolling(window=w, min_periods=max(1, w // 2))
            df[out_name] = ro.apply(fn, args=args, raw=True)
            features.append(out_name)

    return features


def add_threshold_feature(df: pd.DataFrame, column_name: str, thresholds: list[float], out_names: list[str]) -> list[str]:
    """Boolean "did this future-aggregated column cross the threshold" feature — used by the
    ``highlow`` (non-``2``) label generator to turn a max/min future aggregation into a set of
    binary labels at several threshold levels.
    """
    for i, threshold in enumerate(thresholds):
        out_name = out_names[i]
        if threshold > 0.0:
            df[out_name] = df[column_name] >= threshold if abs(threshold) >= 0.75 else df[column_name] <= threshold
        else:
            df[out_name] = df[column_name] <= threshold if abs(threshold) >= 0.75 else df[column_name] >= threshold
    return out_names
