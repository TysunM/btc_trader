"""Small shared type/enum definitions.

Scoped to Binance only per this project's decision to target BTCUSDT spot on Binance —
upstream ITB's ``Venue`` enum also includes ``YAHOO``/``MT5``; those venues (and their
collectors/traders) are intentionally not ported. See the project plan's "Market scope"
decision for why.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum


class Venue(str, Enum):
    BINANCE = "binance"


class AccountBalances:
    """Available assets for trade — updated by trader_binance.py's update_account_balance()."""

    base_quantity: Decimal = Decimal("0")   # e.g. BTC owned, available for trade
    quote_quantity: Decimal = Decimal("0")  # e.g. USDT owned, available for trade
