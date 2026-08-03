"""Periodic OHLC + score-indicator chart sent to Telegram as a PNG. Ported from upstream ITB's
``outputs/notifier_diagram.py``.

matplotlib/seaborn are only imported inside :func:`generate_chart` (matching upstream), so this
module — and the rest of the pipeline — never requires the optional ``diagram`` extra unless a
config actually enables ``diagram_notification_model``.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import requests

from common.model_store import ModelStore
from common.utils import pandas_get_interval
from outputs.notifier_trades import load_all_transactions

log = logging.getLogger("notifier")


async def send_diagram(df: pd.DataFrame, model: dict, config: dict, model_store: ModelStore) -> None:
    freq = config.get("freq")
    time_column = config["time_column"]

    if not ptypes.is_datetime64_any_dtype(df.index):
        if time_column in df.columns:
            df = df.set_index(time_column, drop=False)
        else:
            raise ValueError(f"Neither the index nor column {time_column!r} is datetime-typed.")

    notification_freq = model.get("notification_freq")
    if notification_freq and pandas_get_interval(notification_freq)[0] != pandas_get_interval(freq)[0]:
        return  # only fire when the (longer) diagram interval boundary aligns with this tick

    score_column_names = model.get("score_column_names")
    if isinstance(score_column_names, str):
        score_column_names = [score_column_names]
    elif isinstance(score_column_names, list) and len(score_column_names) > 1:
        score_column_names = [score_column_names[0]]
        log.warning("'score_column_names' should be a single column; only the first will be plotted.")

    score_thresholds = model.get("score_thresholds")
    resampling_freq = model.get("resampling_freq")
    nrows = model.get("nrows")

    vis_columns = ["open", "high", "low", "close", score_column_names[0]]

    for ma in _as_list(model.get("score_ma", [])):
        if not isinstance(ma, int):
            log.error(f"'score_ma' entries must be integers, got {ma!r}. Skipping.")
            continue
        ma_column = f"{score_column_names[0]}_{ma}"
        df[ma_column] = df[score_column_names[0]].rolling(window=ma).mean()
        vis_columns.append(ma_column)
        score_column_names.append(ma_column)

    df_ohlc = resample_ohlc_data(df[vis_columns].reset_index(), resampling_freq, nrows, score_columns=score_column_names)

    df_t = load_all_transactions(config)
    transactions_exist = False
    if df_t is not None and len(df_t) > 0:
        df_t["buy_long"] = df_t["status"] == "BUY"
        df_t["sell_long"] = df_t["status"] == "SELL"
        df_t = df_t[df_t.timestamp >= df_ohlc.timestamp.min()]
        if len(df_t) > 0:
            transactions_exist = True
            df_t = resample_transaction_data(df_t, resampling_freq, 0, "buy_long", "sell_long")
            df_plot = df_ohlc.merge(df_t, how="left", left_on="timestamp", right_on="timestamp")
        else:
            df_plot = df_ohlc
    else:
        df_plot = df_ohlc

    title = config["symbol"]
    if config.get("description"):
        title += ": " + config["description"]

    fig = generate_chart(
        df_plot,
        title,
        buy_signal_column="buy_long" if transactions_exist else None,
        sell_signal_column="sell_long" if transactions_exist else None,
        score_column=score_column_names,
        thresholds=score_thresholds,
    )

    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        img_data = buf.getvalue()

    bot_token = config.get("telegram_bot_token")
    chat_id = config.get("telegram_chat_id")
    if not bot_token or not chat_id:
        log.info("(Telegram not configured) diagram generated but not sent.")
        return

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        requests.post(url=url, data={"chat_id": chat_id, "caption": "", "parse_mode": "markdown"}, files={"photo": img_data}, timeout=15)
    except Exception as e:
        log.error(f"Error sending Telegram diagram: {e}")


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def resample_ohlc_data(df: pd.DataFrame, freq: str, nrows: int | None, score_columns: list[str]) -> pd.DataFrame:
    """Resample to a lower frequency for display (e.g. 1m bars -> 5m for a week-long chart)."""
    ohlc = {"timestamp": "first", "open": "first", "high": "max", "low": "min", "close": "last"}
    for col in score_columns:
        ohlc[col] = lambda x: max(x) if len(x) > 0 and all(x > 0.0) else (min(x) if len(x) > 0 and all(x < 0.0) else np.mean(x))

    df_out = df.resample(freq, on="timestamp").apply(ohlc)
    del df_out["timestamp"]
    df_out.reset_index(inplace=True)
    return df_out.tail(nrows) if nrows else df_out


def resample_transaction_data(df: pd.DataFrame, freq: str, nrows: int, buy_signal_column: str, sell_signal_column: str) -> pd.DataFrame:
    """Collapse arbitrary-timestamp transactions onto the chart's regular time raster (true if
    at least one transaction of that side occurred within the interval)."""
    aggregations = {
        "timestamp": "first",
        buy_signal_column: lambda x: bool((x == True).any()),  # noqa: E712
        sell_signal_column: lambda x: bool((x == True).any()),  # noqa: E712
    }
    df_out = df.resample(freq, on="timestamp").apply(aggregations)
    del df_out["timestamp"]
    df_out.reset_index(inplace=True)
    return df_out.tail(nrows) if nrows else df_out


def generate_chart(df: pd.DataFrame, title: str, buy_signal_column: str | None, sell_signal_column: str | None, score_column, thresholds: list | None):
    """Price (high/low fill + close line, with buy/sell markers) on the left axis, the primary
    score column (+ secondary score columns and threshold lines) on the right axis."""
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt

    import seaborn as sns

    sns.set_style("white", {"axes.grid": False, "axes.spines.top": True, "axes.edgecolor": "lightgrey"})
    sns.set_context("notebook")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    plt.fill_between(df.timestamp, df.low, df.high, step="mid", lw=0.0, facecolor="skyblue", alpha=0.4)
    sns.lineplot(data=df, x="timestamp", y="close", drawstyle="steps-mid", lw=0.5, color="blue", ax=ax1)

    if buy_signal_column and buy_signal_column in df.columns:
        sns.lineplot(data=df[df[buy_signal_column] == True], x="timestamp", y="close", lw=0, markerfacecolor="green", markersize=10, marker="^", alpha=0.6, ax=ax1)  # noqa: E712
    if sell_signal_column and sell_signal_column in df.columns:
        sns.lineplot(data=df[df[sell_signal_column] == True], x="timestamp", y="close", lw=0, markerfacecolor="red", markersize=10, marker="v", alpha=0.6, ax=ax1)  # noqa: E712

    ax1.set(xlabel=None)
    ax1.set_ylabel("Close price", color="blue", fontsize=16)
    ymin, ymax = df["low"].min(), df["high"].max()
    ax1.set(ylim=(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.005))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.tick_params(axis="x", rotation=90)
    ax1.xaxis.grid(True)

    if not isinstance(score_column, list):
        score_column = [score_column]
    main_score_column = score_column[0] if score_column else None

    if main_score_column and main_score_column in df.columns:
        ax2 = ax1.twinx()
        ymax = max(df[main_score_column].abs().max(), max(thresholds) if thresholds else 0.0)
        ax2.set(ylim=(-ymax * 1.2, ymax * 1.2))
        ax2.xaxis.grid(True)
        ax2.axhline(0.0, lw=0.1, color="black")
        for threshold in thresholds or []:
            ax2.axhline(threshold, lw=3.0, color="lightgray")

        for i, sec_col in reversed(list(enumerate(score_column))):
            if i == 0:
                continue
            sns.lineplot(data=df, x="timestamp", y=sec_col, drawstyle="default", lw=i, color="violet", alpha=0.5, ax=ax2)

        sns.lineplot(data=df, x="timestamp", y=main_score_column, drawstyle="steps-mid", lw=1.0, color="red", ax=ax2)
        ax2.set_ylabel("Intelligent Indicator", color="r", fontsize=16)

    plt.title(title, fontsize=14)
    return fig
