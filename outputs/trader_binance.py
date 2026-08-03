"""Real Binance order execution. Ported from upstream ITB's ``outputs/trader_binance.py``.

**This is the highest-risk file in the project.** It is reachable only if ALL of the following
are true (see the README's "Safety / Guardrails" section for the full explanation):

1. A config's ``output_sets`` explicitly includes a ``trader_binance`` entry.
2. The ``ITB_ALLOW_LIVE_TRADING=1`` environment variable is set (checked in
   ``common/generators.py::output_feature_set`` — outside the JSONC config file on purpose, so
   an accidental config edit alone can never enable it).
3. ``service/server.py`` was started with ``--i-understand-live-trading-risk``.

None of this project's shipped configs include a ``trader_binance`` output_set entry, so none of
this code runs by default under any circumstance.

Even with all three gates open, per-call config flags still apply exactly as upstream:
``trade_model.test_order_before_submit``, ``trade_model.simulate_order_execution`` (defaults to
``true`` in this project — a deliberate deviation from upstream's risky-by-default ``false``,
see ``common/config.py``), and ``trade_model.no_trades_only_data_processing``.

Addition not in upstream: every order :func:`execute_order` would submit is first written to a
local dry-run audit log (``orders_dry_run.log``, gitignored), even when fully disabled — so
there's always a "what would have happened" trail to review before ever flipping the switches.

Known upstream gaps, ported faithfully rather than silently fixed here (tracked as v2 items):
no min-notional/lot-size/precision validation beyond the rounding helpers, no idempotency guard
against double-submission, no max-daily-loss circuit breaker, all-in/all-out sizing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from binance.enums import (
    ORDER_STATUS_CANCELED,
    ORDER_STATUS_EXPIRED,
    ORDER_STATUS_FILLED,
    ORDER_STATUS_NEW,
    ORDER_STATUS_PARTIALLY_FILLED,
    ORDER_STATUS_PENDING_CANCEL,
    ORDER_STATUS_REJECTED,
    ORDER_TYPE_LIMIT,
    SIDE_BUY,
    SIDE_SELL,
    TIME_IN_FORCE_GTC,
)

from common.model_store import ModelStore
from common.utils import now_timestamp, round_down_str, round_str, to_decimal
from inputs import collector_binance
from outputs.notifier_trades import get_signal
from service.app_state import get_state

log = logging.getLogger("trader")


async def trader_binance(df, model: dict, config: dict, model_store: ModelStore) -> None:
    """Per-tick order-execution state machine. Only reachable per this module's docstring."""
    state = get_state()
    now_ts = now_timestamp()

    buy_signal_column = model.get("buy_signal_column")
    sell_signal_column = model.get("sell_signal_column")
    signal = get_signal(df, buy_signal_column, sell_signal_column, config)
    signal_side = signal.get("side")
    close_price = signal.get("close_price")

    log.info(f"===> Start trade task. Timestamp {now_ts}.")

    #
    # Sync trade status against any in-flight order
    #
    status = state.status
    if status in ("BUYING", "SELLING"):
        order_status = await update_order_status(config)
        order = state.order

        if not order or not order_status:
            await update_trade_status(config)
            log.error(f"Bad order or order status {order}. Full reset/init needed.")
            return

        if order_status == ORDER_STATUS_FILLED:
            log.info(f"Limit order filled. {order}")
            state.status = "BOUGHT" if status == "BUYING" else "SOLD"
        elif order_status in (ORDER_STATUS_REJECTED, ORDER_STATUS_EXPIRED, ORDER_STATUS_CANCELED):
            log.error(f"Order failed with status {order_status}.")
            state.status = "SOLD" if status == "BUYING" else "BOUGHT"
        elif order_status == ORDER_STATUS_PENDING_CANCEL:
            return
        elif order_status == ORDER_STATUS_PARTIALLY_FILLED:
            pass
        elif order_status == ORDER_STATUS_NEW:
            pass
    elif status not in ("BOUGHT", "SOLD"):
        log.error(f"Unexpected trade status {status!r}.")

    #
    # Kill any order that hasn't filled within one scheduler interval
    #
    status = state.status
    if status in ("BUYING", "SELLING"):
        order_status = await cancel_order(config)
        if not order_status:
            await update_trade_status(config)
            return
        state.status = "SOLD" if status == "BUYING" else "BOUGHT"

    #
    # Act on the current signal
    #
    status = state.status
    await update_account_balance(config)

    if status == "SOLD" and signal_side == "BUY":
        await new_limit_order(config, side=SIDE_BUY)
        if model.get("no_trades_only_data_processing"):
            log.info("no_trades_only_data_processing=true: signal computed but no order state change.")
        else:
            state.status = "BUYING"
    elif status == "BOUGHT" and signal_side == "SELL":
        await new_limit_order(config, side=SIDE_SELL)
        if model.get("no_trades_only_data_processing"):
            log.info("no_trades_only_data_processing=true: signal computed but no order state change.")
        else:
            state.status = "SELLING"

    log.info("<=== End trade task.")


#
# Order and asset status
#

async def update_trade_status(config: dict) -> None:
    """Read account state from Binance and set the local status machine accordingly."""
    state = get_state()
    symbol = config["symbol"]

    try:
        open_orders = collector_binance.client.get_open_orders(symbol=symbol)
    except Exception as e:
        log.error(f"Binance error in get_open_orders: {e}")
        return

    if not open_orders:
        await update_account_balance(config)
        last_kline = state.analyzer.get_last_kline()
        last_close_price = to_decimal(last_kline["close"])

        base_quantity = state.account_info.base_quantity
        base_value_in_quote = base_quantity * last_close_price
        quote_quantity = state.account_info.quote_quantity

        state.status = "SOLD" if quote_quantity >= base_value_in_quote else "BOUGHT"
    elif len(open_orders) == 1:
        order = open_orders[0]
        if order.get("side") == SIDE_SELL:
            state.status = "SELLING"
        elif order.get("side") == SIDE_BUY:
            state.status = "BUYING"
        else:
            log.error(f"Order has neither BUY nor SELL side: {order}")
    else:
        log.error("More than one open order exists. Fix manually -- no auto-recovery for this case.")


async def update_order_status(config: dict) -> str | None:
    """Refresh the current order's status from Binance and return it."""
    state = get_state()
    symbol = config["symbol"]

    order = state.order
    order_id = order.get("orderId", 0) if order else 0
    if not order_id:
        log.error("No current order id to check status for.")
        return None

    try:
        new_order = collector_binance.client.get_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        log.error(f"Binance error in get_order: {e}")
        return None

    if not new_order:
        return None
    order.update(new_order)
    return order["status"]


async def update_account_balance(config: dict) -> None:
    state = get_state()
    try:
        base_balance = collector_binance.client.get_asset_balance(asset=config["base_asset"])
        state.account_info.base_quantity = Decimal(base_balance.get("free", "0"))
    except Exception as e:
        log.error(f"Binance error in get_asset_balance (base): {e}")
        return

    try:
        quote_balance = collector_binance.client.get_asset_balance(asset=config["quote_asset"])
        state.account_info.quote_quantity = Decimal(quote_balance.get("free", "0"))
    except Exception as e:
        log.error(f"Binance error in get_asset_balance (quote): {e}")


#
# Cancel and order creation
#

async def cancel_order(config: dict) -> str | None:
    state = get_state()
    symbol = config["symbol"]

    order = state.order
    order_id = order.get("orderId", 0) if order else 0
    if order_id == 0:
        return None

    try:
        log.info(f"Cancelling order id {order_id}")
        new_order = collector_binance.client.cancel_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        log.error(f"Binance error in cancel_order: {e}")
        return None

    if not new_order:
        return None
    order.update(new_order)
    return order["status"]


async def new_limit_order(config: dict, side: str) -> dict | None:
    """Create a new limit order sized against the full available balance on the relevant side."""
    state = get_state()
    symbol = config["symbol"]
    now_ts = now_timestamp()
    trade_model = config.get("trade_model", {})

    last_kline = state.analyzer.get_last_kline()
    last_close_price = to_decimal(last_kline["close"])
    if not last_close_price:
        log.error("Cannot determine last close price; refusing to create an order.")
        return None

    price_adjustment = trade_model.get("limit_price_adjustment", 0.0)
    if side == SIDE_BUY:
        price = last_close_price * Decimal(1.0 - price_adjustment)
    else:
        price = last_close_price * Decimal(1.0 + price_adjustment)

    price_str = round_str(price, 2)
    price = Decimal(price_str)

    if side == SIDE_BUY:
        quantity = state.account_info.quote_quantity
        percentage = Decimal(trade_model.get("percentage_used_for_trade", 99.0))
        quantity = (quantity * percentage) / Decimal(100.0) / price
    else:
        quantity = state.account_info.base_quantity

    quantity_str = round_down_str(quantity, 6)

    order_spec = dict(
        symbol=symbol,
        side=side,
        type=ORDER_TYPE_LIMIT,
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=quantity_str,
        price=price_str,
    )

    _log_dry_run_order(config, order_spec)

    if trade_model.get("no_trades_only_data_processing"):
        log.info(f"no_trades_only_data_processing=true: not submitting order spec {order_spec}")
        order = None
    else:
        order = execute_order(config, order_spec)

    state.order = order
    state.order_time = now_ts
    return order


def execute_order(config: dict, order: dict) -> dict | None:
    """Validate and submit an order. Gated by trade_model.test_order_before_submit and
    trade_model.simulate_order_execution -- the latter defaults true in this project (see the
    module docstring), unlike upstream's default-false."""
    trade_model = config.get("trade_model", {})

    # TODO (tracked from upstream, not fixed here): no min-notional/lot-size/precision
    # validation against exchange filters beyond the Decimal rounding helpers above.

    if trade_model.get("test_order_before_submit"):
        try:
            log.info(f"Submitting test order: {order}")
            collector_binance.client.create_test_order(**order)
        except Exception as e:
            log.error(f"Binance error in create_test_order: {e}")
            return None

    if trade_model.get("simulate_order_execution", True):
        log.info(f"[SIMULATED] Order not actually submitted: {order}")
        return None

    try:
        log.info(f"Submitting order: {order}")
        submitted = collector_binance.client.create_order(**order)
    except Exception as e:
        log.error(f"Binance error in create_order: {e}")
        return None

    if not submitted or not submitted.get("status"):
        return None
    return submitted


def _log_dry_run_order(config: dict, order_spec: dict) -> None:
    """Always record what would have been submitted, regardless of the simulate/no-trades
    gates above -- a persistent audit trail for reviewing before ever going live."""
    path = Path(config.get("data_folder", ".")) / "orders_dry_run.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+") as f:
            f.write(f"{datetime.now().isoformat()} {order_spec}\n")
    except Exception as e:
        log.error(f"Failed to write dry-run order log: {e}")
