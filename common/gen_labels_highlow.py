"""Label generation: labels are computed from *future* data (as opposed to normal features,
computed from past data), and are only ever used for training — never available online.

``highlow2`` is the generator ITB's own sample configs actually use by default (a simplified
triple-barrier "which side gets touched first" label). ``highlow`` is the older, simpler
generator: a fixed set of "did the max/min future move cross this threshold" booleans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.gen_features import add_threshold_feature
from common.gen_features_rolling_agg import add_future_aggregations
from common.utils import first_location_of_crossing_threshold


def generate_labels_highlow(df: pd.DataFrame, horizon: int) -> list[str]:
    """Fixed set of binary labels comparing the future max-high / min-low (over ``horizon``
    rows) against a fixed grid of thresholds. Mutates ``df`` in place and returns the label
    names — matches upstream's in-place-mutate-and-return-names-only shape (like
    ``generate_features_talib``), unlike ``highlow2``'s ``(df, labels)`` tuple return.
    """
    labels: list[str] = []
    windows = [horizon]

    labels += add_future_aggregations(df, "high", np.max, windows=windows, suffix="_max", rel_column_name="close", rel_factor=100.0)
    high_col = f"high_max_{horizon}"
    labels += add_threshold_feature(df, high_col, thresholds=[1.0, 1.5, 2.0, 2.5, 3.0], out_names=["high_10", "high_15", "high_20", "high_25", "high_30"])
    labels += add_threshold_feature(df, high_col, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], out_names=["high_01", "high_02", "high_03", "high_04", "high_05"])

    labels += add_future_aggregations(df, "low", np.min, windows=windows, suffix="_min", rel_column_name="close", rel_factor=100.0)
    low_col = f"low_min_{horizon}"
    labels += add_threshold_feature(df, low_col, thresholds=[-0.1, -0.2, -0.3, -0.4, -0.5], out_names=["low_01", "low_02", "low_03", "low_04", "low_05"])
    labels += add_threshold_feature(df, low_col, thresholds=[-1.0, -1.5, -2.0, -2.5, -3.0], out_names=["low_10", "low_15", "low_20", "low_25", "low_30"])

    # Ratio of max-high to min-low magnitude, scaled to [-1, +1] (+1: no drawdown; -1: no upside).
    df[high_col] = df[high_col].clip(lower=0)
    df[low_col] = df[low_col].clip(upper=0) * -1
    column_sum = df[high_col] + df[low_col]
    ratio_col = f"high_to_low_{horizon}"
    df[ratio_col] = ((df[high_col] / column_sum) * 2) - 1

    return labels


def generate_labels_highlow2(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Generate one or more boolean "first touch" labels per threshold.

    config: {"columns": [close, high, low], "function": "high"|"low",
             "thresholds": [...], "tolerance": float, "horizon": int, "names": [...]}
    """
    column_names = config.get("columns")
    close_column, high_column, low_column = column_names

    function = config.get("function")
    if function not in ("high", "low"):
        raise ValueError(f"Unknown function name {function!r}. Only 'high' or 'low' are possible.")

    tolerance = config.get("tolerance")

    thresholds = config.get("thresholds")
    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    if function == "high":
        thresholds = [abs(t) for t in thresholds]
        price_columns = [high_column, low_column]
    else:
        thresholds = [-abs(t) for t in thresholds]
        price_columns = [low_column, high_column]

    tolerances = [round(-t * tolerance, 6) for t in thresholds]

    horizon = config.get("horizon")
    names = config.get("names")
    if len(names) != len(thresholds):
        raise ValueError("'highlow2' label generator: each threshold needs exactly one name.")

    labels = []
    for i, threshold in enumerate(thresholds):
        first_cross_labels(df, horizon, [threshold, tolerances[i]], close_column, price_columns, names[i])
        labels.append(names[i])

    return df, labels


def first_cross_labels(
    df: pd.DataFrame, horizon: int, thresholds: list[float], close_column: str, price_columns: list[str], out_column: str
) -> str:
    """True if the price crosses ``thresholds[0]`` before it crosses ``thresholds[1]`` in the
    opposite direction (a simplified triple-barrier: target barrier vs. invalidation barrier,
    with ``horizon`` acting as the time barrier).
    """
    df["_first_idx"] = first_location_of_crossing_threshold(df, horizon, thresholds[0], close_column, price_columns[0])
    df["_second_idx"] = first_location_of_crossing_threshold(df, horizon, thresholds[1], close_column, price_columns[1])

    def is_high_true(x: np.ndarray) -> bool:
        if np.isnan(x[0]):
            return False
        if np.isnan(x[1]):
            return True
        return x[0] <= x[1]

    df[out_column] = df[["_first_idx", "_second_idx"]].apply(is_high_true, raw=True, axis=1)
    df.drop(columns=["_first_idx", "_second_idx"], inplace=True)
    return out_column
