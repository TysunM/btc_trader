"""Shared helpers: decimal rounding, time-raster arithmetic, data-source merging, dynamic
generator resolution, and score computation.

Ported from ITB's ``common/utils.py``. One deliberate change from upstream: this module does
not import ``common.gen_features`` (upstream has ``from common.gen_features import *`` here,
which is a circular import — ``gen_features.py`` in turn imports ``from common.utils import *``
— and nothing in this module actually calls a ``gen_features`` function; it only worked upstream
because of a specific import order threaded through ``service/App.py``). Dropping it makes the
dependency graph acyclic without changing behavior.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any, Callable

import dateparser
import numpy as np
import pandas as pd
import pytz
from apscheduler.triggers.cron import CronTrigger
from sklearn import metrics

# --- Decimals (order price/quantity rounding — correctness-critical for any real order) ---


def to_decimal(value) -> Decimal:
    """Convert to a Decimal with 8-digit precision, truncating (never rounding up)."""
    n = 8
    quantum = Decimal(1) / (Decimal(10) ** n)
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_DOWN)


def round_str(value, digits: int) -> str:
    quantum = Decimal(1) / (Decimal(10) ** digits)
    ret = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{ret:.{digits}f}"


def round_down_str(value, digits: int) -> str:
    """Round toward zero — used for order quantity so we never request more than the
    account actually holds due to floating-point/precision rounding up."""
    quantum = Decimal(1) / (Decimal(10) ** digits)
    ret = Decimal(str(value)).quantize(quantum, rounding=ROUND_DOWN)
    return f"{ret:.{digits}f}"


# --- Interval / time-raster arithmetic (pandas frequency <-> wall-clock) ---


def pandas_interval_length_ms(freq: str) -> int:
    return int(pd.Timedelta(freq).to_pytimedelta().total_seconds() * 1000)


def pandas_get_interval(freq: str, timestamp: int | datetime | None = None) -> tuple[int, int]:
    """Find the discrete interval containing ``timestamp`` (or now) and return (start_ms, end_ms)."""
    if timestamp is None:
        ts = int(datetime.now(timezone.utc).timestamp())
    elif isinstance(timestamp, datetime):
        ts = int(timestamp.replace(tzinfo=timezone.utc).timestamp())
    elif isinstance(timestamp, int):
        ts = timestamp
    else:
        raise ValueError(f"Cannot convert timestamp {timestamp!r} of type {type(timestamp)}")

    interval_length_sec = pandas_interval_length_ms(freq) / 1000
    start = (ts // interval_length_sec) * interval_length_sec
    end = start + interval_length_sec
    return int(start * 1000), int(end * 1000)


def get_interval_count_from_start_dt(freq: str, start_dt: datetime) -> int:
    """How many whole intervals lie between ``start_dt`` and now, plus a small safety margin."""
    interval_length_td = pd.Timedelta(freq).to_pytimedelta()
    now = datetime.now(timezone.utc)
    interval_count = (now - start_dt) // interval_length_td
    return interval_count + 2


def get_start_dt_for_interval_count(freq: str, interval_count: int) -> datetime:
    interval_length_td = pd.Timedelta(freq).to_pytimedelta()
    period_length_td = interval_length_td * (interval_count + 1)
    return datetime.now(timezone.utc) - period_length_td


def freq_to_cron_trigger(freq: str) -> CronTrigger:
    """Map a pandas frequency to an APScheduler cron trigger, firing shortly after each bar
    closes (not exactly on the boundary) so the exchange has published the closed candle."""
    if freq.endswith("min"):
        n = freq[:-3]
        return CronTrigger(minute="*" if n == "1" else f"*/{n}", second="1", timezone="UTC")
    if freq.endswith("h"):
        n = freq[:-1]
        return CronTrigger(hour="*" if n == "1" else f"*/{n}", minute="0", second="20", timezone="UTC")
    if freq.endswith("D"):
        n = freq[:-1]
        return CronTrigger(day="*" if n == "1" else f"*/{n}", hour="0", minute="1", second="0", timezone="UTC")
    if freq.endswith("W"):
        n = freq[:-1]
        return CronTrigger(
            week="*" if n == "1" else f"*/{n}", day_of_week="1", hour="1", minute="0", second="0", timezone="UTC"
        )
    if freq.endswith("MS"):
        n = freq[:-2]
        return CronTrigger(
            month="*" if n == "1" else f"*/{n}", day="1", hour="1", minute="0", second="0", timezone="UTC"
        )
    raise ValueError(f"Cannot convert frequency '{freq}' to a cron trigger.")


def now_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def find_index(df: pd.DataFrame, date_str: str, column_name: str = "timestamp") -> int:
    """Return the row index of the record matching ``date_str`` in ``column_name``."""
    d = dateparser.parse(date_str)
    try:
        res = df[df[column_name] == d]
    except TypeError:
        if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
            d = d.replace(tzinfo=pytz.utc)
        else:
            d = d.replace(tzinfo=None)
        res = df[df[column_name] == d]

    if res is None or len(res) == 0:
        raise ValueError(f"Cannot find date '{date_str}' in column '{column_name}'.")
    return res.index[0]


def notnull_tail_rows(df: pd.DataFrame) -> int:
    """Maximum number of tail rows with no NaN in any column — used online to find how far
    back predictions can validly be recomputed."""
    nan_df = df.isnull()
    nan_cols = nan_df.any()
    nan_cols = nan_cols[nan_cols].index.to_list()
    if not nan_cols:
        return len(df)
    tail_rows = nan_df[nan_cols].values[::-1].argmax(axis=0).min()
    return int(tail_rows)


# --- Dynamic generator resolution (the plugin extension point) ---


def resolve_generator_name(gen_name: str) -> Callable | None:
    """Resolve ``'module.path:function_name'`` to a function reference, or None if unresolvable.

    This is the extension point: any custom feature/label/signal/output generator is just a
    Python function with the conventional ``fn(df, config, global_config, model_store)``
    signature, referenced from config by this dotted-colon name — no registration step needed.
    """
    mod_and_func = gen_name.split(":", 1)
    if len(mod_and_func) < 2:
        return None
    mod_name, func_name = mod_and_func

    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return None
    return getattr(mod, func_name, None)


def find_algorithm_by_name(algorithms: list[dict], name: str) -> dict:
    return next(x for x in algorithms if x.get("name") == name)


# --- Data merging ---


def merge_data_sources(data_sources: list[dict], time_column: str, freq: str, merge_interpolate: bool) -> pd.DataFrame:
    """Merge multiple source DataFrames (each in ``ds['df']``) onto one common time raster."""
    for ds in data_sources:
        df = ds.get("df")

        if time_column in df.columns:
            df = df.set_index(time_column)
        elif df.index.name != time_column:
            raise ValueError(f"Data source {ds.get('folder')!r} is missing the time column {time_column!r}.")

        if ds.get("column_prefix"):
            prefix = ds["column_prefix"]
            df.columns = [c if c.startswith(prefix + "_") else f"{prefix}_{c}" for c in df.columns]

        ds["start"] = df.first_valid_index()
        ds["end"] = df.last_valid_index()
        ds["df"] = df

    range_start = min(ds["start"] for ds in data_sources)
    range_end = min(ds["end"] for ds in data_sources)

    index = pd.date_range(range_start, range_end, freq=freq)
    df_out = pd.DataFrame(index=index)
    df_out.index.name = time_column
    df_out.insert(0, time_column, df_out.index)

    for ds in data_sources:
        df_out = df_out.join(ds["df"])

    if merge_interpolate:
        num_columns = df_out.select_dtypes((float, int)).columns.tolist()
        for col in num_columns:
            df_out[col] = df_out[col].interpolate()

    return df_out


def append_df_drop_concat(df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Append ``new_df`` to ``df``, dropping any overlapping tail rows from ``df`` first so the
    new (presumably more current) values win in the overlap range. Used by the online
    :class:`~common.analyzer.Analyzer` to merge freshly fetched klines into its rolling window.
    """
    if len(new_df) == 0:
        return df
    df_wo_overlap = df[: new_df.index[0]].iloc[:-1]
    return pd.concat([df_wo_overlap, new_df])


# --- Scoring ---


def compute_scores(y_true: pd.Series, y_hat: pd.Series) -> dict[str, float]:
    """Classification metrics for a [0,1] score column vs. a boolean/int label."""
    y_true = y_true.astype(int)
    y_hat_class = np.where(y_hat.values > 0.5, 1, 0)

    try:
        auc = metrics.roc_auc_score(y_true, y_hat.fillna(value=0))
    except ValueError:
        auc = 0.0
    try:
        ap = metrics.average_precision_score(y_true, y_hat.fillna(value=0))
    except ValueError:
        ap = 0.0

    scores = dict(
        auc=auc,
        ap=ap,
        f1=metrics.f1_score(y_true, y_hat_class),
        precision=metrics.precision_score(y_true, y_hat_class),
        recall=metrics.recall_score(y_true, y_hat_class),
    )
    return {k: round(float(v), 3) for k, v in scores.items()}


def compute_scores_regression(y_true: pd.Series, y_hat: pd.Series) -> dict[str, float]:
    try:
        mae = metrics.mean_absolute_error(y_true, y_hat)
    except ValueError:
        mae = np.nan
    try:
        mape = metrics.mean_absolute_percentage_error(y_true, y_hat)
    except ValueError:
        mape = np.nan
    try:
        r2 = metrics.r2_score(y_true, y_hat)
    except ValueError:
        r2 = np.nan

    y_true_class = np.where(y_true.values > 0.0, 1, -1)
    y_hat_class = np.where(y_hat.values > 0.0, 1, -1)
    try:
        auc = metrics.roc_auc_score(y_true_class, y_hat_class)
    except ValueError:
        auc = 0.0
    try:
        ap = metrics.average_precision_score(y_true_class, y_hat_class)
    except ValueError:
        ap = 0.0

    scores = dict(
        mae=mae,
        mape=mape,
        r2=r2,
        auc=auc,
        ap=ap,
        f1=metrics.f1_score(y_true_class, y_hat_class),
        precision=metrics.precision_score(y_true_class, y_hat_class),
        recall=metrics.recall_score(y_true_class, y_hat_class),
    )
    return {k: round(float(v), 3) for k, v in scores.items()}


def first_location_of_crossing_threshold(
    df: pd.DataFrame, horizon: int, threshold: float, close_column_name: str, price_column_name: str
) -> pd.Series:
    """For each row, find the (relative) offset to the first future row where
    ``price_column_name`` crosses ``threshold`` percent away from that row's close.

    Positive ``threshold`` searches for an increase (typically against the ``high`` column),
    negative searches for a decrease (typically against ``low``). Returns NaN where the price
    never crosses within ``horizon`` rows. Used by the ``highlow2`` label generator to implement
    a triple-barrier-style "which side gets touched first" label.
    """

    def fn_high(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        p = x[0, 0]
        p_threshold = p * (1 + threshold / 100.0)
        idx = np.argmax(x[1:, 1] > p_threshold)
        if idx == 0 and x[1, 1] <= p_threshold:
            return np.nan
        return idx

    def fn_low(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        p = x[0, 0]
        p_threshold = p * (1 + threshold / 100.0)
        idx = np.argmax(x[1:, 1] < p_threshold)
        if idx == 0 and x[1, 1] >= p_threshold:
            return np.nan
        return idx

    rl = df[[close_column_name, price_column_name]].rolling(horizon + 1, min_periods=(horizon // 2), method="table")

    if threshold > 0:
        df_out = rl.apply(fn_high, raw=True, engine="numba")
    elif threshold < 0:
        df_out = rl.apply(fn_low, raw=True, engine="numba")
    else:
        raise ValueError("Threshold cannot be zero.")

    df_out = df_out.shift(-horizon)
    return df_out.iloc[:, 0]
