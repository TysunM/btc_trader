"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """A small, deterministic synthetic OHLCV DataFrame with an upward-then-downward price
    path, hourly frequency, UTC DatetimeIndex — enough rows for rolling-window features with
    small windows (<=20) to produce non-NaN values well before the end of the series.
    """
    n = 200
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")

    # Smooth trend + small noise so rolling stats/slopes are well-defined and non-degenerate.
    trend = np.concatenate([np.linspace(100, 150, n // 2), np.linspace(150, 120, n - n // 2)])
    noise = rng.normal(0, 0.5, n)
    close = trend + noise
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, n)
    volume = rng.uniform(10, 100, n)
    trades = rng.integers(50, 500, n)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trades": trades,
            "tb_base_av": volume * rng.uniform(0.3, 0.7, n),
        },
        index=idx,
    )


@pytest.fixture
def base_config(tmp_path) -> dict:
    """A minimal but structurally complete config dict, rooted at a pytest tmp_path so tests
    never touch the real data/ tree."""
    return {
        "train": True,
        "venue": "binance",
        "time_column": "timestamp",
        "data_folder": str(tmp_path / "data"),
        "symbol": "TESTUSDT",
        "freq": "1h",
        "label_horizon": 5,
        "features_horizon": 20,
        "train_length": 0,
        "predict_length": 20,
        "model_folder": "MODELS",
        "merge_file_name": "data.csv",
        "feature_file_name": "features.csv",
        "matrix_file_name": "matrix.csv",
        "predict_file_name": "predictions.csv",
        "signal_file_name": "signals.csv",
        "signal_models_file_name": "signal_models",
        "data_sources": [{"folder": "TESTUSDT", "file": "klines", "column_prefix": ""}],
        "feature_sets": [],
        "label_sets": [],
        "train_features": [],
        "labels": [],
        "algorithms": [],
        "train_feature_sets": [],
        "signal_sets": [],
        "output_sets": [],
    }
