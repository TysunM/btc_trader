"""Signal generation: turns a small number of point-wise ML scores into a final buy/sell
decision via configurable, backtestable rules (as opposed to more ML).
"""

from __future__ import annotations

import pandas as pd


def generate_smoothen_scores(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Row-wise average of the specified columns, optional binarization, optional moving average."""
    columns = config.get("columns")
    if not columns:
        raise ValueError(f"'columns' must be a non-empty string/list, got {columns!r}")
    if isinstance(columns, str):
        columns = [columns]

    out_column = df[columns].mean(skipna=True, axis=1)

    point_threshold = config.get("point_threshold")
    if point_threshold:
        out_column = out_column >= point_threshold

    window = config.get("window")
    if isinstance(window, int):
        out_column = out_column.rolling(window, min_periods=window // 2).mean()
    elif isinstance(window, float):
        out_column = out_column.ewm(span=window, min_periods=int(window) // 2, adjust=False).mean()

    names = config.get("names")
    if not isinstance(names, str):
        raise ValueError(f"'names' must be a non-empty string, got {names!r}")
    df[names] = out_column
    return df, [names]


def combine_scores_relative(df: pd.DataFrame, buy_column: str, sell_column: str, out_column: str) -> pd.Series:
    """Mutual adjustment: if buy and sell scores are equally high, output is 0. In [-1, +1]."""
    buy_plus_sell = df[buy_column] + df[sell_column]
    score = ((df[buy_column] / buy_plus_sell) * 2) - 1.0
    df[out_column] = score
    return score


def combine_scores_difference(df: pd.DataFrame, buy_column: str, sell_column: str, out_column: str) -> pd.Series:
    """How much higher the buy score is than the sell score. Positive => buy, negative => sell."""
    score = df[buy_column] - df[sell_column]
    df[out_column] = score
    return score


def generate_combine_scores(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Combine a (buy_score, sell_score) pair — each in [0,1] — into one signed score."""
    columns = config.get("columns")
    if not columns or not isinstance(columns, list) or len(columns) != 2:
        raise ValueError(f"'columns' must be a 2-element list [buy_col, sell_col], got {columns!r}")
    up_column, down_column = columns
    out_column = config.get("names")

    combine = config.get("combine")
    if combine == "relative":
        combine_scores_relative(df, up_column, down_column, out_column)
    elif combine == "difference":
        combine_scores_difference(df, up_column, down_column, out_column)
    else:
        df[out_column] = df[[up_column, down_column]].apply(
            lambda x: x[0] if x[0] >= x[1] else -x[1], raw=True, axis=1
        )

    if config.get("coefficient"):
        df[out_column] = df[out_column] * config["coefficient"]
    if config.get("constant"):
        df[out_column] = df[out_column] + config["constant"]

    return df, [out_column]


def generate_threshold_rule(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """One score column, two thresholds -> boolean buy/sell signal columns."""
    parameters = config.get("parameters", {})
    columns = config.get("columns")
    if not columns:
        raise ValueError(f"'columns' must be a non-empty string, got {columns!r}")
    if isinstance(columns, list):
        columns = columns

    buy_signal_column, sell_signal_column = config.get("names")
    df[buy_signal_column] = df[columns] >= parameters.get("buy_signal_threshold")
    df[sell_signal_column] = df[columns] <= parameters.get("sell_signal_threshold")
    return df, [buy_signal_column, sell_signal_column]


def generate_threshold_rule2(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Two score columns, each with its own threshold — both must agree for a signal."""
    parameters = config.get("parameters", {})
    columns = config.get("columns")
    if not columns or not isinstance(columns, list) or len(columns) != 2:
        raise ValueError(f"'columns' must be a 2-element list, got {columns!r}")
    score_column, score_column_2 = columns

    buy_signal_column, sell_signal_column = config.get("names")

    df[buy_signal_column] = (df[score_column] >= parameters.get("buy_signal_threshold")) & (
        df[score_column_2] >= parameters.get("buy_signal_threshold_2")
    )
    df[sell_signal_column] = (df[score_column] <= parameters.get("sell_signal_threshold")) & (
        df[score_column_2] <= parameters.get("sell_signal_threshold_2")
    )
    return df, [buy_signal_column, sell_signal_column]
