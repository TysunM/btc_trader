"""Venue dispatch for data collectors. Binance-only (see common.types.Venue)."""

from __future__ import annotations

from common.types import Venue


def get_download_functions(venue: Venue):
    """Return the bulk/offline downloader function for the given venue."""
    if venue == Venue.BINANCE:
        from inputs.collector_binance import download_klines

        return download_klines
    raise ValueError(f"Unsupported venue for download: {venue!r}")


def get_collector_functions(venue: Venue):
    """Return (fetch_klines, health_check) online collector functions for the given venue."""
    if venue == Venue.BINANCE:
        from inputs.collector_binance import fetch_klines, health_check

        return fetch_klines, health_check
    raise ValueError(f"Unsupported venue for online collection: {venue!r}")
