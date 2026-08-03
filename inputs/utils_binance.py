"""Binance-specific helpers. Ported from upstream ITB's ``inputs/utils_binance.py``."""

from __future__ import annotations


def binance_freq_from_pandas(freq: str) -> str:
    """Map a pandas frequency string (e.g. ``1h``, ``1min``) to a Binance kline interval (e.g.
    ``1h``, ``1m``)."""
    if freq.endswith("min"):
        freq = freq.replace("min", "m")
    elif freq.endswith("D"):
        freq = freq.replace("D", "d")
    elif freq.endswith("W"):
        freq = freq.replace("W", "w")
    elif freq == "BMS":
        freq = freq.replace("BMS", "M")

    if len(freq) == 1:
        freq = "1" + freq

    if not (2 <= len(freq) <= 3) or not freq[:-1].isdigit() or freq[-1] not in ("m", "h", "d", "w", "M"):
        raise ValueError(f"Unsupported Binance frequency derived from {freq!r}.")

    return freq
