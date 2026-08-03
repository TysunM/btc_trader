"""Regime filter overlay (v2 quant pass, item 4 — see CHANGELOG_V2.md).

``scripts/simulate.py``'s grid search fits one fixed threshold set across the entire backtest
window regardless of trend/volatility regime — a classic overfit-to-whichever-regime-dominated-
the-window failure mode. A threshold tuned during a strong trend can whipsaw badly once the
market chops sideways.

This is a ``signal_sets`` generator using the project's existing zero-registration plugin
mechanism (``common.utils.resolve_generator_name``, dispatched via ``module:function`` in
config) — no changes to ``common/generators.py``'s core dispatch were needed. It must run
*after* the ``threshold_rule``/``threshold_rule2`` entry in ``signal_sets`` (since it reads and
overwrites the buy/sell signal columns those generators produce), gating buy (entry) signals to
only fire when ADX indicates a trending regime. Sell (exit) signals are *not* gated by default —
you should always be able to exit a position regardless of regime; only new entries are filtered.
"""

from __future__ import annotations

import pandas as pd

from common.model_store import ModelStore
from common.ta_adapter import adx


def generate_regime_gate(df: pd.DataFrame, config: dict, global_config: dict, model_store: ModelStore):
    """config: {
        "close_column": "close" (default), "high_column": "high", "low_column": "low",
        "adx_window": 14 (default), "adx_threshold": 20.0 (default — ADX below this is
            conventionally considered "no trend" / choppy),
        "buy_signal_column": "buy_signal_column", "sell_signal_column": "sell_signal_column",
        "gate_sell": false (default) -- if true, also gates exits, not just entries,
        "gate_column_name": "regime_ok" (default) -- also emitted for visibility/debugging,
    }

    Overwrites the buy (and optionally sell) signal columns in place, ANDing them with the
    regime gate. Must appear after the threshold_rule entry that produces those columns.
    """
    close_col = config.get("close_column", "close")
    high_col = config.get("high_column", "high")
    low_col = config.get("low_column", "low")
    window = config.get("adx_window", 14)
    threshold = config.get("adx_threshold", 20.0)
    buy_col = config.get("buy_signal_column", "buy_signal_column")
    sell_col = config.get("sell_signal_column", "sell_signal_column")
    gate_sell = config.get("gate_sell", False)
    gate_name = config.get("gate_column_name", "regime_ok")

    for required in (close_col, high_col, low_col, buy_col, sell_col):
        if required not in df.columns:
            raise ValueError(
                f"generate_regime_gate: column {required!r} not found. This generator must run "
                f"after the feature/signal generators that produce OHLC and buy/sell columns."
            )

    regime_ok = (adx(df[high_col], df[low_col], df[close_col], window) >= threshold).fillna(False)

    df[gate_name] = regime_ok
    df[buy_col] = df[buy_col] & regime_ok
    if gate_sell:
        df[sell_col] = df[sell_col] & regime_ok

    return df, [gate_name, buy_col, sell_col] if gate_sell else [gate_name, buy_col]
