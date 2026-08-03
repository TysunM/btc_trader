"""Fee and slippage modeling for backtest scoring (v2 quant pass, item 1 — see CHANGELOG_V2.md).

Upstream ITB's ``common.backtesting.simulated_trade_performance`` scores raw price deltas with
no execution-cost modeling at all, which inflates the apparent profitability of high-frequency
threshold configs in ``scripts/simulate.py``'s grid search. This module is opt-in (default
``fee_bps=0, slippage_bps=0`` reproduces the v1 baseline exactly) and composable: one function
adjusts a single leg's effective fill price, called from ``common/backtesting.py``.
"""

from __future__ import annotations

import numpy as np

# Binance's default (non-BNB-discounted) spot taker fee is 0.10% per side. Maker is typically
# the same or lower; we use taker as the conservative default since threshold-crossing signals
# in this pipeline are directional market-order-style entries, not resting limit orders.
DEFAULT_TAKER_FEE_BPS = 10.0

# A conservative placeholder for spread + market impact on a liquid pair like BTCUSDT. Real
# slippage depends on order size and book depth, neither of which this simulator models — treat
# this as "better than assuming zero," not a precise estimate.
DEFAULT_SLIPPAGE_BPS = 2.0


def adjusted_execution_price(price: float, side: str, fee_bps: float, slippage_bps: float) -> float:
    """Effective fill price for one BUY or SELL leg after fee + slippage.

    BUY: you pay more than the quoted price (both fee and adverse slippage push cost up).
    SELL: you receive less than the quoted price (both push proceeds down).

    Correct for a position-tracking backtest that knows which side each leg is. NOT used by
    ``common.backtesting.simulated_trade_performance`` — that function's long/short buckets
    compare consecutive same-side executions against each other (not a literal buy-then-sell
    pair), so adjusting both readings in the same direction mostly cancels out rather than
    compounding. Use :func:`round_trip_cost_pct` for that algorithm instead; this function is
    kept for other v2 modules (and any future backtest engine) that do track sides explicitly.
    """
    total_bps = (fee_bps + slippage_bps) / 10_000.0
    side = side.upper()
    if side == "BUY":
        return price * (1.0 + total_bps)
    if side == "SELL":
        return price * (1.0 - total_bps)
    raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")


def round_trip_cost_pct(fee_bps: float, slippage_bps: float) -> float:
    """Round-trip (entry + exit) cost as a percentage, deducted once per simulated transaction
    in ``common.backtesting.simulated_trade_performance``. Simpler and more robust than trying
    to adjust individual execution prices within that function's self-referential long/short
    bucketing (see :func:`adjusted_execution_price`'s docstring) — every discrete transaction
    is charged one full round-trip's worth of cost, which is the standard convention for
    per-trade cost modeling and can't accidentally cancel out between legs.
    """
    return 2.0 * (fee_bps + slippage_bps) / 100.0


def sharpe_like_ratio(trade_returns_pct: list[float]) -> float:
    """Per-trade-return-volatility-adjusted score for ``scripts/simulate.py``'s
    ``simulate_model.rank_by: "sharpe"`` option (v2 item 7, CHANGELOG_V2.md).

    Plain mean/std of the trade-level percent returns — not annualized, since these are
    irregularly-spaced discrete trade events rather than a fixed-period return series. Guards
    against an undefined ratio for 0 or 1 trades, or a degenerate zero-variance sequence.
    """
    if not trade_returns_pct:
        return 0.0
    arr = np.asarray(trade_returns_pct, dtype=float)
    if len(arr) < 2 or np.std(arr) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr))
