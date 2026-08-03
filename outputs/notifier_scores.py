"""Telegram score-band notifier: sends a message only when the trade score crosses into a new
"band" (configured thresholds) or a configured time interval elapses — throttling to avoid
spamming on every tick. Ported from upstream ITB's ``outputs/notifier_scores.py``.

Sends via raw Telegram Bot API HTTP calls (``requests``), matching upstream's actual behavior —
not the ``aiogram`` framework upstream's own README aspirationally lists under "external
integrations" but never actually uses.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas.api.types as ptypes
import requests

from common.model_store import ModelStore

log = logging.getLogger("notifier")


async def send_score_notification(df: pd.DataFrame, model: dict, config: dict, model_store: ModelStore) -> None:
    symbol = config["symbol"]
    freq = config["freq"]
    time_column = config["time_column"]

    score_column_names = model.get("score_column_names")
    if not score_column_names:
        log.error("score_notification_model requires a non-empty 'score_column_names' list.")
        return

    row = df.iloc[-1]
    interval_length = pd.Timedelta(freq).to_pytimedelta()

    if ptypes.is_datetime64_any_dtype(df.index):
        close_time = row.name
    elif time_column in df.columns and ptypes.is_datetime64_any_dtype(df[time_column]):
        close_time = row[time_column]
    else:
        raise ValueError(f"Neither the index nor column {time_column!r} is datetime-typed.")
    close_time += interval_length

    close_price = row["close"]
    trade_scores = [row[col] for col in score_column_names]
    trade_score_primary = trade_scores[0]
    trade_score_secondary = trade_scores[1] if len(trade_scores) > 1 else None

    band_no, band = _find_score_band(trade_score_primary, model)

    # model dict persists across ticks (it's the same output_sets config entry object each
    # time), so storing prev_band_no directly on it is how upstream tracks state without a
    # separate server object -- kept as-is since this notifier has no other cross-tick state.
    prev_band_no = model.get("prev_band_no")
    if prev_band_no is not None:
        band_up = abs(band_no) > abs(prev_band_no)
        band_dn = abs(band_no) < abs(prev_band_no)
    else:
        band_up = True
        band_dn = True
    model["prev_band_no"] = band_no

    if band and band.get("frequency"):
        new_to_time_interval = close_time.minute % band["frequency"] == 0
    else:
        new_to_time_interval = False

    notification_is_needed = (
        (model.get("notify_band_up") and band_up)
        or (model.get("notify_band_dn") and band_dn)
        or new_to_time_interval
    )
    if not notification_is_needed:
        return

    symbol_char = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}.get(symbol, symbol)
    band_change_char = "^" if band_up else ("v" if band_dn else "")

    primary_score_str = f"{trade_score_primary:+.2f} {band_change_char} "
    secondary_score_str = f"{trade_score_secondary:+.2f}" if trade_score_secondary is not None else ""

    if band:
        message = f"{band.get('sign', '')} {symbol_char} {close_price:,.0f} Indicator: {primary_score_str} {secondary_score_str} {band.get('text', '')} {freq}"
        if band.get("bold"):
            message = "*" + message + "*"
    else:
        message = f"{symbol_char} {close_price:,.0f} Indicator: {primary_score_str} {secondary_score_str} {freq}"

    message = message.replace("+", "%2B")

    bot_token = config.get("telegram_bot_token")
    chat_id = config.get("telegram_chat_id")
    if not bot_token or not chat_id:
        log.info(f"(Telegram not configured) {message}")
        return

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=markdown&text={message}"
        response = requests.get(url, timeout=10)
        if not response.json().get("ok"):
            log.error(f"Telegram API returned an error response: {response.text}")
    except Exception as e:
        log.error(f"Error sending Telegram notification: {e}")


def _find_score_band(score_value: float, model: dict) -> tuple[int, dict | None]:
    """Find which configured band ``score_value`` falls into.

    Positive band numbers (1, 2, ...) mean the score is above a positive threshold; negative
    band numbers (-1, -2, ...) mean it's below a negative threshold; 0 means neutral (no band).
    """
    bands = sorted(model.get("positive_bands", []), key=lambda x: x.get("edge"), reverse=True)
    band_no, band = next(((i, x) for i, x in enumerate(bands) if score_value >= x.get("edge")), (len(bands), None))
    band_no = len(bands) - band_no

    if not band:
        bands = sorted(model.get("negative_bands", []), key=lambda x: x.get("edge"))
        band_no, band = next(((i, x) for i, x in enumerate(bands) if score_value < x.get("edge")), (len(bands), None))
        band_no = -(len(bands) - band_no)

    return band_no, band
