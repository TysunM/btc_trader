"""Paper-trading simulator and shared trade-signal/transaction-log helpers.

``trader_simulation`` never places a real order — it just tracks a simulated BUY/SELL state
machine and logs to ``transactions.txt``, purely for observing what the signals *would* have
done. ``get_signal``/``get_transaction_path`` are also used by ``outputs/trader_binance.py``
(the real, gated trader).

Ported from upstream ITB's ``outputs/notifier_trades.py``, adapted to read/write trade state via
``service.app_state.get_state()`` instead of the global ``App`` class, and to take ``config``
explicitly wherever upstream reached for ``App.config``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes
import requests

from common.model_store import ModelStore
from common.ta_adapter import atr
from service.app_state import get_state
from v2.position_sizing import atr_position_size, check_stop_take

log = logging.getLogger("notifier")


async def trader_simulation(df: pd.DataFrame, model: dict, config: dict, model_store: ModelStore) -> None:
    try:
        transaction = await generate_trader_transaction(df, model, config)
    except Exception as e:
        log.error(f"Error in trader_simulation: {e}")
        return
    if not transaction:
        return

    try:
        await send_transaction_message(transaction, config)
    except Exception as e:
        log.error(f"Error in send_transaction_message: {e}")


async def generate_trader_transaction(df: pd.DataFrame, model: dict, config: dict) -> dict | None:
    """Simple all-in/all-out simulated strategy: flip fully long or fully short on each signal.

    v2 item 5 (CHANGELOG_V2.md), both opt-in via ``trade_model``:
    - ``stop_loss_pct``/``take_profit_pct``: if the current open (BUY) position's unrealized
      return crosses either threshold, force an exit this tick regardless of the model's own
      sell signal. Fully enforced (not just advisory).
    - ``position_sizing.enabled``: log an ATR-based recommended position size fraction alongside
      each BUY. Advisory only — see this module's docstring for why the simulator itself still
      tracks all-in/all-out rather than a fractional equity curve.
    """
    state = get_state()
    transaction_path = get_transaction_path(config)
    trade_model = config.get("trade_model", {})

    buy_signal_column = model.get("buy_signal_column")
    sell_signal_column = model.get("sell_signal_column")

    signal = get_signal(df, buy_signal_column, sell_signal_column, config)
    signal_side = signal.get("side")
    close_price = signal.get("close_price")
    close_time = signal.get("close_time")

    if not state.transaction:
        t_status, t_price = None, None
    else:
        t_status = state.transaction.get("status")
        t_price = state.transaction.get("price")

    # Stop-loss / take-profit: enforced ahead of (and independent of) the model's own signal.
    exit_reason = None
    if t_status == "BUY" and t_price:
        exit_reason = check_stop_take(
            entry_price=t_price,
            current_price=close_price,
            stop_loss_pct=trade_model.get("stop_loss_pct"),
            take_profit_pct=trade_model.get("take_profit_pct"),
        )

    if exit_reason:
        profit = close_price - t_price
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="SELL")
        log.warning(f"{exit_reason.upper()} triggered: entry={t_price:.2f} exit={close_price:.2f}")
    elif signal_side == "BUY" and (not t_status or t_status == "SELL"):
        profit = t_price - close_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="BUY")
        _log_position_size_advisory(df, close_price, trade_model)
    elif signal_side == "SELL" and (not t_status or t_status == "BUY"):
        profit = close_price - t_price if t_price else 0.0
        t_dict = dict(timestamp=str(close_time), price=close_price, profit=profit, status="SELL")
    else:
        return None

    state.transaction = t_dict
    # File schema intentionally unchanged from upstream (timestamp,price,profit,status) --
    # load_last_transaction/load_all_transactions/generate_transaction_stats all assume exactly
    # these 4 columns. Exit reason and sizing advisories go to the log/Telegram message only.
    with open(transaction_path, "a+") as f:
        f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in t_dict.values()) + "\n")

    log.info(f"Trade simulator transaction: {t_dict}")
    return t_dict


def _log_position_size_advisory(df: pd.DataFrame, close_price: float, trade_model: dict) -> None:
    """v2 item 5: log (not enforce) an ATR-based recommended position size for this entry."""
    sizing_config = trade_model.get("position_sizing", {})
    if not sizing_config.get("enabled"):
        return
    try:
        atr_series = atr(df["high"], df["low"], df["close"], window=sizing_config.get("atr_window", 14))
        atr_value = atr_series.iloc[-1]
        equity = sizing_config.get("nominal_equity", 10_000.0)
        fraction = atr_position_size(
            equity=equity,
            atr=atr_value,
            price=close_price,
            risk_pct=sizing_config.get("risk_pct", 1.0),
            kelly_cap=sizing_config.get("kelly_cap", 0.25),
        )
        log.info(
            f"Position sizing advisory: ATR={atr_value:.2f} -> recommended {fraction * 100:.1f}% "
            f"of a ${equity:,.0f} nominal account (not enforced by the paper simulator)."
        )
    except Exception as e:
        log.error(f"Error computing position sizing advisory: {e}")


async def send_transaction_message(transaction: dict, config: dict) -> None:
    profit, profit_percent, profit_descr, profit_percent_descr = await generate_transaction_stats(config)

    if transaction.get("status") == "SELL":
        message = "SOLD: "
    elif transaction.get("status") == "BUY":
        message = "BOUGHT: "
    else:
        log.error("Unexpected transaction status.")
        return

    message += f" Profit: {profit_percent:.2f}% {profit:.4f}"

    bot_token = config.get("telegram_bot_token")
    chat_id = config.get("telegram_chat_id")
    if not bot_token or not chat_id:
        log.info(f"(Telegram not configured) {message}")
        return

    _send_telegram_message(bot_token, chat_id, message)

    if transaction.get("status") == "SELL":
        message = "LONG transactions stats (4 weeks)\n"
    elif transaction.get("status") == "BUY":
        message = "SHORT transactions stats (4 weeks)\n"
    else:
        return

    message += f"sum={profit_percent_descr['count'] * profit_percent_descr['mean']:.2f}% count={int(profit_percent_descr['count'])}\n"
    message += f"mean={profit_percent_descr['mean']:.4f}% std={profit_percent_descr['std']:.4f}%\n"
    message += f"min={profit_percent_descr['min']:.4f}% median={profit_percent_descr['50%']:.4f}% max={profit_percent_descr['max']:.4f}%\n"

    _send_telegram_message(bot_token, chat_id, message)


def _send_telegram_message(bot_token: str, chat_id: str, message: str) -> None:
    message = message.replace("+", "%2B")
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=markdown&text={message}"
        response = requests.get(url, timeout=10)
        if not response.json().get("ok"):
            log.error(f"Telegram API returned an error response: {response.text}")
    except Exception as e:
        log.error(f"Error sending Telegram notification: {e}")


async def generate_transaction_stats(config: dict):
    """Compute stats (profit, mean/std/min/median/max) over the last 4 weeks of transactions,
    assuming the latest transaction was just appended to the transaction log."""
    transaction_path = get_transaction_path(config)

    df = pd.read_csv(transaction_path, parse_dates=[0], header=None, names=["timestamp", "close", "profit", "status"], date_format="ISO8601")

    mask = df["timestamp"] >= (datetime.now() - timedelta(weeks=4))
    df = df[max(mask.idxmax() - 1, 0):]  # keep one prior row to compute the first relative profit

    df["prev_close"] = df["close"].shift()
    df["profit_percent"] = df.apply(lambda x: (100.0 * x["profit"] / x["prev_close"]) if x["prev_close"] else 0.0, axis=1)
    df = df.iloc[1:]

    long_df = df[df["status"] == "SELL"]
    short_df = df[df["status"] == "BUY"]

    last_transaction = df.iloc[-1]
    profit = last_transaction["profit"]
    profit_percent = last_transaction["profit_percent"]

    df2 = long_df if last_transaction["status"] == "SELL" else short_df

    profit_descr = df2["profit"].describe()
    profit_percent_descr = df2["profit_percent"].describe()

    return profit, profit_percent, profit_descr, profit_percent_descr


def get_signal(df: pd.DataFrame, buy_signal_column: str, sell_signal_column: str, config: dict) -> dict:
    """From the last row of ``df``, produce the current trade signal (side/price/time)."""
    freq = config["freq"]
    row = df.iloc[-1]
    interval_length = pd.Timedelta(freq).to_pytimedelta()

    if not ptypes.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime for get_signal().")
    close_time = row.name + interval_length  # timestamp marks interval start; add the interval to get its end

    close_price = row["close"]
    buy_signal = row[buy_signal_column]
    sell_signal = row[sell_signal_column]

    if buy_signal and sell_signal:
        signal_side = "BOTH"  # should not normally happen
    elif buy_signal:
        signal_side = "BUY"
    elif sell_signal:
        signal_side = "SELL"
    else:
        signal_side = ""

    return {"side": signal_side, "close_price": close_price, "close_time": close_time}


def load_last_transaction(config: dict) -> dict:
    transaction_path = get_transaction_path(config)

    t_dict = dict(timestamp=str(datetime.now()), price=0.0, profit=0.0, status="")
    if transaction_path.is_file():
        line = ""
        with open(transaction_path) as f:
            for line in f:
                pass
        if line:
            t_dict = dict(zip("timestamp,price,profit,status".split(","), line.strip().split(",")))
            t_dict["timestamp"] = pd.to_datetime(t_dict["timestamp"], utc=True)
            t_dict["price"] = float(t_dict["price"])
            t_dict["profit"] = float(t_dict["profit"])
    else:
        transaction_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transaction_path, "a+") as f:
            f.write("2020-01-01 00:00:00,0.0,0.0,SELL\n")
    return t_dict


def load_all_transactions(config: dict) -> pd.DataFrame | None:
    transaction_path = get_transaction_path(config)
    if not transaction_path.is_file():
        log.warning(f"Transaction file does not exist: {transaction_path}")
        return None
    df = pd.read_csv(transaction_path, names=["timestamp", "price", "profit", "status"], header=None)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df.astype({"timestamp": "datetime64[ns, UTC]", "price": "float64", "profit": "float64", "status": "str"})


def get_transaction_path(config: dict) -> Path:
    return Path(config["data_folder"]) / config["symbol"] / "transactions.txt"
