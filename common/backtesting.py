"""Backtesting and trade performance via simple trade simulation.

Ported from upstream ITB's ``common/backtesting.py``, with one v2 addition (CHANGELOG_V2.md
item 1): optional ``fee_bps``/``slippage_bps`` parameters, both defaulting to ``0.0`` so the v1
baseline (tag ``v1.0-itb-port``) is reproduced exactly when unset. When set, a flat round-trip
cost (``v2.fees.round_trip_cost_pct``) is deducted from every simulated transaction's profit —
deliberately visible as an explicit before/after rather than silently baked into the original
fee-less numbers.

Note on why costs are a flat per-transaction deduction rather than a per-leg execution-price
adjustment: this function's long/short buckets compare consecutive *same-side* executions
against each other (see the loop below), not a literal buy-then-sell pair. Adjusting both
readings' prices in the same direction would make the fee mostly cancel out between legs instead
of compounding — see ``v2/fees.py``'s docstring for the full explanation. A flat deduction avoids
that trap entirely.
"""

from __future__ import annotations

import pandas as pd

from v2.fees import round_trip_cost_pct


def simulated_trade_performance(
    df: pd.DataFrame,
    buy_signal_column: str,
    sell_signal_column: str,
    price_column: str,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
):
    """Simulate alternating buy/sell trades following the boolean signal columns and return
    (overall, long, short) performance dicts with transaction counts, profit, and profit ratios.

    ``fee_bps``/``slippage_bps`` (v2, opt-in): a round-trip cost, computed once via
    :func:`v2.fees.round_trip_cost_pct`, deducted from every completed transaction's profit.
    0/0 reproduces the v1 baseline exactly.
    """
    is_buy_mode = True
    cost_pct = round_trip_cost_pct(fee_bps, slippage_bps) if (fee_bps or slippage_bps) else 0.0

    long_profit = 0.0
    long_profit_percent = 0.0
    long_transactions = 0
    long_profitable = 0
    long_trade_returns_pct: list[float] = []

    short_profit = 0.0
    short_profit_percent = 0.0
    short_transactions = 0
    short_profitable = 0
    short_trade_returns_pct: list[float] = []

    last_short_price = 0.0
    last_long_price = 0.0

    df = df[[sell_signal_column, buy_signal_column, price_column]]
    for _, sell_signal, buy_signal, price in df.itertuples(name=None):
        if not price or pd.isnull(price):
            continue
        if is_buy_mode:
            if buy_signal:
                previous_price = last_short_price
                profit = (previous_price - price) if previous_price > 0 else 0.0
                profit_percent = 100.0 * profit / previous_price if previous_price > 0 else 0.0
                if previous_price > 0 and cost_pct:
                    profit -= previous_price * cost_pct / 100.0
                    profit_percent -= cost_pct
                short_profit += profit
                short_profit_percent += profit_percent
                short_transactions += 1
                if profit > 0:
                    short_profitable += 1
                if previous_price > 0:
                    short_trade_returns_pct.append(profit_percent)
                last_short_price = price
                is_buy_mode = False
        else:
            if sell_signal:
                previous_price = last_long_price
                profit = (price - previous_price) if previous_price > 0 else 0.0
                profit_percent = 100.0 * profit / previous_price if previous_price > 0 else 0.0
                if previous_price > 0 and cost_pct:
                    profit -= previous_price * cost_pct / 100.0
                    profit_percent -= cost_pct
                long_profit += profit
                long_profit_percent += profit_percent
                long_transactions += 1
                if profit > 0:
                    long_profitable += 1
                if previous_price > 0:
                    long_trade_returns_pct.append(profit_percent)
                last_long_price = price
                is_buy_mode = True

    long_performance = {
        "#transactions": long_transactions,
        "profit": round(long_profit, 2),
        "%profit": round(long_profit_percent, 1),
        "#profitable": long_profitable,
        "%profitable": round(100.0 * long_profitable / long_transactions, 1) if long_transactions else 0.0,
        "profit/T": round(long_profit / long_transactions, 2) if long_transactions else 0.0,
        "%profit/T": round(long_profit_percent / long_transactions, 1) if long_transactions else 0.0,
        "_trade_returns_pct": long_trade_returns_pct,
    }

    short_performance = {
        "#transactions": short_transactions,
        "profit": round(short_profit, 2),
        "%profit": round(short_profit_percent, 1),
        "#profitable": short_profitable,
        "%profitable": round(100.0 * short_profitable / short_transactions, 1) if short_transactions else 0.0,
        "profit/T": round(short_profit / short_transactions, 2) if short_transactions else 0.0,
        "%profit/T": round(short_profit_percent / short_transactions, 1) if short_transactions else 0.0,
        "_trade_returns_pct": short_trade_returns_pct,
    }

    profit = long_profit + short_profit
    profit_percent = long_profit_percent + short_profit_percent
    transaction_no = long_transactions + short_transactions
    profitable = (long_profitable + short_profitable) / transaction_no if transaction_no else 0.0

    performance = {
        "#transactions": transaction_no,
        "profit": profit,
        "%profit": profit_percent,
        "profitable": profitable,
        "profitable_percent": round(100.0 * profitable / transaction_no, 1) if transaction_no else 0.0,
        "profit/T": round(profit / transaction_no, 2) if transaction_no else 0.0,
        "%profit/T": round(profit_percent / transaction_no, 1) if transaction_no else 0.0,
        "_trade_returns_pct": long_trade_returns_pct + short_trade_returns_pct,
    }

    return performance, long_performance, short_performance
