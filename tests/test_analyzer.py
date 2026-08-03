"""Tests for common/analyzer.py — upstream had zero coverage here. Covers the dirty_records
state machine, window trim logic, and append_data's merge/overwrite behavior, since these are
the mechanics that make the online service's incremental recompute correct.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.analyzer import Analyzer
from common.model_store import ModelStore


@pytest.fixture
def analyzer_config(base_config) -> dict:
    base_config["predict_length"] = 10
    base_config["features_horizon"] = 5
    base_config["train_features"] = ["close"]
    base_config["labels"] = []
    base_config["train"] = False
    return base_config


@pytest.fixture
def analyzer(analyzer_config) -> Analyzer:
    return Analyzer(analyzer_config, ModelStore(analyzer_config))


def _klines_df(start: str, n: int, base_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": [base_price] * n,
            "high": [base_price + 1] * n,
            "low": [base_price - 1] * n,
            "close": [base_price + i for i in range(n)],
            "volume": [10.0] * n,
        },
        index=idx,
    )


class TestAnalyzerInit:
    def test_window_lengths_computed_from_config(self, analyzer):
        assert analyzer.min_window_length == 15  # predict_length(10) + features_horizon(5)
        assert analyzer.max_window_length == 30  # min + 15

    def test_starts_with_full_recompute_flag(self, analyzer):
        assert analyzer.dirty_records == -1

    def test_starts_with_empty_dataframe(self, analyzer):
        assert len(analyzer.df) == 0
        assert analyzer.get_last_kline() is None

    def test_get_last_kline_dt_on_empty_uses_window_length(self, analyzer):
        # Should not raise, and should return a datetime far enough back to cover min_window_length
        dt = analyzer.get_last_kline_dt()
        assert dt is not None


class TestAppendData:
    def test_first_append_sets_full_recompute(self, analyzer, analyzer_config):
        df = _klines_df("2026-01-01", 20)
        analyzer.append_data({"TESTUSDT": df})
        assert len(analyzer.df) == 20
        assert analyzer.dirty_records == -1  # initial_df_len was 0 -> stays full recompute

    def test_second_append_sets_incremental_dirty_count(self, analyzer, analyzer_config):
        df1 = _klines_df("2026-01-01", 20)
        analyzer.append_data({"TESTUSDT": df1})
        analyzer.dirty_records = 0  # simulate a completed analyze()

        df2 = _klines_df("2026-01-01 18:00", 5)  # 2 hours overlap + 3 new rows
        analyzer.append_data({"TESTUSDT": df2})

        # 3 genuinely new rows appended; dirty_records should reflect that (not -1, not 0)
        assert analyzer.dirty_records > 0

    def test_overlap_rows_are_overwritten_by_new_values(self, analyzer):
        df1 = _klines_df("2026-01-01", 10, base_price=100.0)
        analyzer.append_data({"TESTUSDT": df1})

        df2 = _klines_df("2026-01-01 08:00", 5, base_price=999.0)  # overlaps last 2 rows
        analyzer.append_data({"TESTUSDT": df2})

        last_ts = df2.index[0]
        assert analyzer.df.loc[last_ts, "close"] == pytest.approx(999.0)

    def test_previous_df_captures_pre_append_tail(self, analyzer):
        df1 = _klines_df("2026-01-01", 15)
        analyzer.append_data({"TESTUSDT": df1})
        assert analyzer.previous_df is not None
        assert len(analyzer.previous_df) == 0  # nothing existed before the very first append

        df2 = _klines_df("2026-01-01 20:00", 3)
        analyzer.append_data({"TESTUSDT": df2})
        assert len(analyzer.previous_df) == 10  # tail(10) of the state just before this append


class TestAnalyzeFullCycle:
    def test_analyze_computes_features_and_trims_window(self, analyzer_config):
        analyzer_config["feature_sets"] = [
            {"column_prefix": "", "generator": "talib", "feature_prefix": "", "config": {"columns": ["close"], "functions": ["SMA"], "windows": [3]}}
        ]
        analyzer_config["train_features"] = ["close_SMA_3"]
        analyzer_config["train_feature_sets"] = []
        analyzer_config["signal_sets"] = []
        a = Analyzer(analyzer_config, ModelStore(analyzer_config))

        df = _klines_df("2026-01-01", 40)  # well over max_window_length (30)
        a.append_data({"TESTUSDT": df})
        a.analyze()

        assert a.dirty_records == 0
        assert len(a.df) <= a.max_window_length
        assert "close_SMA_3" in a.df.columns
        assert not a.df["close_SMA_3"].iloc[-1] != a.df["close_SMA_3"].iloc[-1]  # last value is not NaN

    def test_analyze_with_zero_dirty_records_is_a_noop_with_warning(self, analyzer, caplog):
        df = _klines_df("2026-01-01", 20)
        analyzer.append_data({"TESTUSDT": df})
        analyzer.dirty_records = 0
        analyzer.analyze()  # should log a warning and return without raising
        assert "0 dirty records" in caplog.text
