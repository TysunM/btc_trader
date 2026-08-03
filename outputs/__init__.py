"""Venue dispatch for trader-related callables. Binance-only (see common.types.Venue)."""

from __future__ import annotations

from common.types import Venue


def get_trader_functions(venue: Venue) -> dict[str, callable]:
    """Return the four trader-related callables for the given venue.

    Example: ``get_trader_functions(Venue.BINANCE)["trader"](df, model, config, model_store)``.
    """
    if venue == Venue.BINANCE:
        from outputs.trader_binance import (
            trader_binance,
            update_account_balance,
            update_order_status,
            update_trade_status,
        )

        return {
            "trader": trader_binance,
            "update_account_balance": update_account_balance,
            "update_order_status": update_order_status,
            "update_trade_status": update_trade_status,
        }
    raise ValueError(f"Unsupported venue for trading: {venue!r}")
