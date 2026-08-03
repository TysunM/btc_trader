"""Volatility-targeted position sizing + stop-loss/take-profit (v2 quant pass, item 5 — see
CHANGELOG_V2.md).

Upstream (and this project's v1 port) sizes all-in/all-out: buy with ``percentage_used_for_trade``
(99%) of quote balance, sell 100% of base balance, with no stop-loss, take-profit, or max-position
anywhere. This module adds two independent, opt-in pieces:

1. :func:`atr_position_size` — ATR-based sizing with a capped Kelly-like fraction ceiling. Wired
   into ``outputs/notifier_trades.py``'s paper simulator as an *advisory* figure surfaced in the
   transaction log — the simulator itself remains a simple all-cash-flips-side model (rewriting
   it into a true fractional-equity portfolio tracker is a larger change than this pass scopes
   in; the sizing recommendation is real and computed correctly, just not yet enforced against a
   tracked equity curve). Real enforcement matters once (and if) ``trader_binance.py`` is ever
   actually enabled — see the README's "Safety" section.
2. :func:`check_stop_take` — an immediate, enforced stop-loss/take-profit check against the
   currently open paper position, independent of the model's own buy/sell signal. This one *is*
   fully enforced: the paper simulator will force an exit the tick a configured threshold is
   crossed, regardless of what the signal columns say.
"""

from __future__ import annotations


def atr_position_size(equity: float, atr: float, price: float, risk_pct: float = 1.0, kelly_cap: float = 0.25) -> float:
    """Fraction of equity to allocate to a position, sized so that a 1-ATR adverse move costs
    approximately ``risk_pct`` percent of equity — the standard "risk a fixed percent per unit
    of volatility" sizing rule. Capped at ``kelly_cap`` so a very small ATR (a quiet market)
    doesn't imply an unreasonably large, effectively unlimited position.
    """
    if atr <= 0 or price <= 0 or equity <= 0:
        return 0.0
    risk_amount = equity * (risk_pct / 100.0)
    units = risk_amount / atr
    fraction = (units * price) / equity
    return min(fraction, kelly_cap)


def check_stop_take(entry_price: float, current_price: float, stop_loss_pct: float | None, take_profit_pct: float | None) -> str | None:
    """Check a long position's unrealized return against configured stop-loss/take-profit
    thresholds. Returns ``"stop_loss"``, ``"take_profit"``, or ``None``. Long-only, matching
    this project's paper simulator and (gated, real) trader — both only ever hold BOUGHT or
    SOLD/flat, never short.
    """
    if entry_price <= 0:
        return None
    change_pct = 100.0 * (current_price - entry_price) / entry_price
    if stop_loss_pct and change_pct <= -abs(stop_loss_pct):
        return "stop_loss"
    if take_profit_pct and change_pct >= abs(take_profit_pct):
        return "take_profit"
    return None
