"""Top/bottom (local extremum) label generators. Ported from upstream ITB's
``common/gen_labels_topbot.py``.

A "top" or "bottom" label is defined by two parameters: ``level`` (the minimum required jump up
or down, present on both sides, for a point to qualify as an extremum) and ``tolerance`` (how
wide a neighborhood around the extremum is also labeled true).
"""

from __future__ import annotations

import pandas as pd


def generate_labels_topbot2(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    init_column_number = len(df.columns)

    column_name = config.get("columns")
    if not isinstance(column_name, str) or not column_name:
        raise ValueError(f"'columns' must be a non-empty string, got {column_name!r}")
    if column_name not in df.columns:
        raise ValueError(f"{column_name!r} not found in input data.")

    function = config.get("function")
    if function not in ("top", "bot"):
        raise ValueError(f"Unknown function {function!r}. Only 'top' or 'bot' are possible.")

    tolerances = config.get("tolerances")
    if not isinstance(tolerances, list):
        tolerances = [tolerances]

    level = config.get("level")
    level = abs(level) if function == "top" else -abs(level)

    names = config.get("names")
    if len(names) != len(tolerances):
        raise ValueError("'topbot2' label generator: each tolerance needs exactly one name.")

    for i, tolerance in enumerate(tolerances):
        df, _ = add_extremum_features(
            df, column_name=column_name, level_fracs=[level], tolerance_frac=abs(level) * tolerance, out_names=names[i:i + 1]
        )

    labels = df.columns.to_list()[init_column_number:]
    return df, labels


def generate_labels_topbot(df: pd.DataFrame, column_name: str, top_level_fracs: list[float], bot_level_fracs: list[float]) -> tuple[pd.DataFrame, list[str]]:
    """Compute top/bottom extremum labels at a fixed grid of tolerances (0.25% .. 3%), each
    producing 5 top and 5 bottom label columns (one per level fraction)."""
    init_column_number = len(df.columns)

    for tolerance_frac, suffix in [
        (0.0025, "025"), (0.005, "05"), (0.0075, "075"), (0.01, "1"), (0.0125, "125"),
        (0.015, "15"), (0.0175, "175"), (0.02, "2"), (0.025, "25"), (0.03, "3"),
    ]:
        top_labels = [f"top{i}_{suffix}" for i in range(1, 6)]
        bot_labels = [f"bot{i}_{suffix}" for i in range(1, 6)]
        df, _ = add_extremum_features(df, column_name=column_name, level_fracs=top_level_fracs, tolerance_frac=tolerance_frac, out_names=top_labels)
        df, _ = add_extremum_features(df, column_name=column_name, level_fracs=bot_level_fracs, tolerance_frac=tolerance_frac, out_names=bot_labels)

    labels = df.columns.to_list()[init_column_number:]
    return df, labels


def add_extremum_features(df: pd.DataFrame, column_name: str, level_fracs: list[float], tolerance_frac: float, out_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    column = df[column_name]
    out_columns = []
    for i, level_frac in enumerate(level_fracs):
        if level_frac > 0.0:
            extrems = find_all_extremums(column, True, level_frac, tolerance_frac)
        else:
            extrems = find_all_extremums(column, False, -level_frac, tolerance_frac)

        out_name = out_names[i]
        out_column = pd.Series(data=False, index=df.index, dtype=bool, name=out_name)
        for extr in extrems:
            # extr = (left_level, left_tolerance, extremum, right_tolerance, right_level)
            out_column.loc[extr[1] + 1: extr[3] - 1] = True
        out_columns.append(out_column)

    df = pd.concat([df] + out_columns, axis=1)
    return df, out_names


def find_all_extremums(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> list[tuple]:
    """Recursively split the series into sub-intervals around each qualifying extremum."""
    extremums = []
    intervals = [(sr.index[0], sr.index[-1] + 1)]

    while intervals:
        interval = intervals.pop()
        extremum = find_one_extremum(sr.loc[interval[0]: interval[1]], is_max, level_frac, tolerance_frac)

        if extremum[0] is not None and extremum[-1] is not None:
            extremums.append(extremum)

        if extremum[0] is not None and interval[0] < extremum[0]:
            intervals.append((interval[0], extremum[0]))
        if extremum[-1] is not None and extremum[-1] < interval[1]:
            intervals.append((extremum[-1], interval[1]))

    return sorted(extremums, key=lambda x: x[2])


def find_one_extremum(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> tuple:
    """Find the extremum of ``sr`` and its level/tolerance interval boundaries, if within ``sr``."""
    if is_max:
        extr_idx = sr.idxmax()
        extr_val = sr.loc[extr_idx]
        level_val = extr_val - level_frac * abs(extr_val)
        tolerance_val = extr_val - tolerance_frac * abs(extr_val)
    else:
        extr_idx = sr.idxmin()
        extr_val = sr.loc[extr_idx]
        level_val = extr_val + level_frac * abs(extr_val)
        tolerance_val = extr_val + tolerance_frac * abs(extr_val)

    sr_left = sr.loc[:extr_idx]
    sr_right = sr.loc[extr_idx:]

    left_level_idx = _left_level_idx(sr_left, is_max, level_val)
    right_level_idx = _right_level_idx(sr_right, is_max, level_val)
    left_tol_idx = _left_level_idx(sr_left, is_max, tolerance_val)
    right_tol_idx = _right_level_idx(sr_right, is_max, tolerance_val)

    return left_level_idx, left_tol_idx, extr_idx, right_tol_idx, right_level_idx


def _left_level_idx(sr_left: pd.Series, is_max: bool, level_val: float):
    matched = sr_left[sr_left < level_val] if is_max else sr_left[sr_left > level_val]
    return matched.index[-1] if len(matched) > 0 else None


def _right_level_idx(sr_right: pd.Series, is_max: bool, level_val: float):
    matched = sr_right[sr_right < level_val] if is_max else sr_right[sr_right > level_val]
    return matched.index[0] if len(matched) > 0 else None
