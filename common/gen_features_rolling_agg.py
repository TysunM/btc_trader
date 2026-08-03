"""Rolling-aggregation helpers shared by the ``itblib``/``itbstats`` feature generators and the
``highlow`` (non-``2``) label generator. Ported from upstream ITB's
``common/gen_features_rolling_agg.py``.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


def add_past_weighted_aggregations(
    df, column_name: str, weight_column_name: str | None, fn, windows, suffix=None,
    rel_column_name: str | None = None, rel_factor: float = 1.0, last_rows: int = 0,
) -> list[str]:
    return _add_weighted_aggregations(df, False, column_name, weight_column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def add_past_aggregations(
    df, column_name: str, fn, windows, suffix=None, rel_column_name: str | None = None,
    rel_factor: float = 1.0, last_rows: int = 0,
) -> list[str]:
    return _add_aggregations(df, False, column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def add_future_aggregations(
    df, column_name: str, fn, windows, suffix=None, rel_column_name: str | None = None,
    rel_factor: float = 1.0, last_rows: int = 0,
) -> list[str]:
    return _add_aggregations(df, True, column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def _add_aggregations(
    df, is_future: bool, column_name: str, fn, windows: Union[int, list[int]], suffix=None,
    rel_column_name: str | None = None, rel_factor: float = 1.0, last_rows: int = 0,
) -> list[str]:
    """Moving aggregation over past or future values of a column. Past windows include the
    current value; future windows do not. Result columns are named
    ``<column><suffix>_<window>``; if a ``rel_column_name`` is given the result is a relative
    (percent) change against that column instead of an absolute value.
    """
    column = df[column_name]
    if isinstance(windows, int):
        windows = [windows]
    rel_column = df[rel_column_name] if rel_column_name else None
    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        if not last_rows:
            feature = column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
        else:
            feature = _aggregate_last_rows(column, w, last_rows, fn)

        if is_future:
            feature = feature.shift(periods=-w)

        feature_name = f"{column_name}{suffix}_{w}"
        features.append(feature_name)
        if rel_column is not None:
            df[feature_name] = rel_factor * (feature - rel_column) / rel_column
        else:
            df[feature_name] = rel_factor * feature

    return features


def _add_weighted_aggregations(
    df, is_future: bool, column_name: str, weight_column_name: str | None, fn, windows,
    suffix=None, rel_column_name: str | None = None, rel_factor: float = 1.0, last_rows: int = 0,
) -> list[str]:
    """Weighted rolling aggregation (typically np.sum -> area-under-the-curve with weights)."""
    column = df[column_name]
    weight_column = df[weight_column_name] if weight_column_name else pd.Series(data=1.0, index=column.index)
    products_column = column * weight_column

    if isinstance(windows, int):
        windows = [windows]
    rel_column = df[rel_column_name] if rel_column_name else None
    if suffix is None:
        suffix = "_" + fn.__name__

    features = []
    for w in windows:
        if not last_rows:
            feature = products_column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
            weights = weight_column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
        else:
            feature = _aggregate_last_rows(products_column, w, last_rows, fn)
            weights = _aggregate_last_rows(weight_column, w, last_rows, fn)

        feature = feature / weights
        if is_future:
            feature = feature.shift(periods=-w)

        feature_name = f"{column_name}{suffix}_{w}"
        features.append(feature_name)
        if rel_column is not None:
            df[feature_name] = rel_factor * (feature - rel_column) / rel_column
        else:
            df[feature_name] = rel_factor * feature

    return features


def area_fn(x: np.ndarray, is_future: bool) -> float:
    level = x[0] if is_future else x[-1]
    x_diff = x - level
    a = np.nansum(x_diff)
    b = np.nansum(np.absolute(x_diff))
    pos = (b + a) / 2
    ratio = pos / b
    return (ratio * 2) - 1


def add_area_ratio(df, is_future: bool, column_name: str, windows, suffix=None, last_rows: int = 0) -> list[str]:
    """Area under/over the current (or, for future, the oldest) element within the window,
    scaled to [-1, +1]."""
    column = df[column_name]
    if isinstance(windows, int):
        windows = [windows]
    if suffix is None:
        suffix = "_area_ratio"

    features = []
    for w in windows:
        if not last_rows:
            feature = column.rolling(window=w, min_periods=max(1, w // 2)).apply(area_fn, kwargs=dict(is_future=is_future), raw=True)
        else:
            feature = _aggregate_last_rows(column, w, last_rows, area_fn, is_future)

        feature_name = f"{column_name}{suffix}_{w}"
        df[feature_name] = feature.shift(periods=-(w - 1)) if is_future else feature
        features.append(feature_name)

    return features


def slope_fn(x: np.ndarray) -> float:
    """OLS slope of x against its integer index — a per-window linear trend."""
    x_idx = np.asarray(range(len(x)))
    y = x
    if np.isnan(y).any():
        mask = ~np.isnan(y)
        x_idx, y = x_idx[mask], y[mask]
    slope, *_ = stats.linregress(x_idx, y)
    return slope


def add_linear_trends(df, is_future: bool, column_name: str, windows, suffix=None, last_rows: int = 0) -> list[str]:
    column = df[column_name]
    if isinstance(windows, int):
        windows = [windows]
    if suffix is None:
        suffix = "_trend"

    features = []
    for w in windows:
        if not last_rows:
            feature = column.rolling(window=w, min_periods=max(1, w // 2)).apply(slope_fn, raw=True)
        else:
            feature = _aggregate_last_rows(column, w, last_rows, slope_fn)

        feature_name = f"{column_name}{suffix}_{w}"
        df[feature_name] = feature.shift(periods=-(w - 1)) if is_future else feature
        features.append(feature_name)

    return features


def to_log_diff(sr: pd.Series) -> pd.Series:
    return np.log(sr).diff()


def to_diff(sr: pd.Series) -> pd.Series:
    """Percent change between consecutive values."""

    def diff_fn(x: np.ndarray) -> float:
        return 100 * (x[1] - x[0]) / x[0]

    return sr.rolling(window=2, min_periods=2).apply(diff_fn, raw=True)


def _aggregate_last_rows(column: pd.Series, window: int, last_rows: int, fn, *args) -> pd.Series:
    """Rolling aggregation computed only for the last ``last_rows`` rows — the online
    incremental-compute fast path (Phase 3)."""
    length = len(column)
    values = [fn(column.iloc[-window - r: length - r].to_numpy(), *args) for r in range(last_rows)]
    feature = pd.Series(data=np.nan, index=column.index, dtype=float)
    feature.iloc[-last_rows:] = list(reversed(values))
    return feature
