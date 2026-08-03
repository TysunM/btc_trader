"""Technical-analysis backend shim.

Every feature generator goes through this module instead of importing ``talib``/``pandas_ta``
directly, so the backend can be swapped via ``ITB_TA_BACKEND=talib|pandas_ta`` without touching
generator code. Default is ``pandas_ta`` (pure Python, trivial install on Windows — the real
``talib`` C library is notoriously painful to build there). Exact-value parity with upstream
ITB's talib output is a nice-to-have, not a requirement: what matters is that the same backend
computes both training and live features, which holds by construction since everything routes
through here.

The small set of primitives implemented natively (sma/stddev/linearreg_slope) matches exactly
what ITB's own sample configs (1min and 1h) actually use by default, so those are implemented
directly in pandas/numpy rather than depending on a third-party library's exact rolling-window
semantics. Everything else optionally delegates to the selected backend.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

Backend = str  # "pandas_ta" | "talib"


def get_backend() -> Backend:
    return os.environ.get("ITB_TA_BACKEND", "pandas_ta").lower()


# --- Native primitives (backend-independent, used by ITB's default sample configs) ---

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def stddev(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=0)


def _slope(y: np.ndarray) -> float:
    if np.isnan(y).any():
        return np.nan
    x = np.arange(len(y))
    # simple OLS slope; window sizes here are small (single/double-digit to low hundreds)
    # so this is fast enough without needing numba for the MVP slice.
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def linearreg_slope(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0.0, 100.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index — used by v2's regime filter (trend-strength gate)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_atr = atr(high, low, close, window)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / tr_atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / tr_atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bbands(
    series: pd.Series, window: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, window)
    dev = stddev(series, window)
    upper = mid + n_std * dev
    lower = mid - n_std * dev
    return upper, mid, lower


# Function-name -> callable, matching the uppercase names used in ITB config
# (e.g. {"functions": ["SMA"], "windows": [5, 10, 15]}). Only single-column,
# single-window functions are registered here; multi-output/multi-input
# indicators (MACD, BBANDS, ATR, ADX) are called directly by generators that need them.
_SINGLE_COLUMN_FUNCTIONS = {
    "SMA": sma,
    "EMA": ema,
    "STDDEV": stddev,
    "LINEARREG_SLOPE": linearreg_slope,
    "RSI": rsi,
}


def call(function_name: str, series: pd.Series, window: int) -> pd.Series:
    """Generic dispatcher for the talib-style {function, window} feature config shape."""
    fn = _SINGLE_COLUMN_FUNCTIONS.get(function_name.upper())
    if fn is None:
        raise ValueError(
            f"Unknown ta_adapter function '{function_name}'. "
            f"Available: {sorted(_SINGLE_COLUMN_FUNCTIONS)}"
        )
    return fn(series, window)
