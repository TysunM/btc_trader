"""Tests for common/utils.py, focused on the order-quantity/price rounding helpers (the
best-covered area upstream, and directly relevant to real-money correctness if trader_binance.py
is ever enabled) plus the merge and generator-resolution helpers.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from common.utils import (
    append_df_drop_concat,
    find_algorithm_by_name,
    merge_data_sources,
    resolve_generator_name,
    round_down_str,
    round_str,
    to_decimal,
)


class TestRounding:
    def test_round_str_rounds_half_up(self):
        assert round_str(1.005, 2) == "1.01"
        assert round_str(1.004, 2) == "1.00"

    def test_round_str_pads_trailing_zeros(self):
        assert round_str(1, 4) == "1.0000"

    def test_round_down_str_truncates_never_rounds_up(self):
        # 0.1999999 at 4 digits must truncate to 0.1999, never round up to 0.2000 -- rounding a
        # sell quantity UP could request more than the account actually holds.
        assert round_down_str(0.19999999, 4) == "0.1999"

    def test_round_down_str_exact_value_unaffected(self):
        assert round_down_str(1.5, 2) == "1.50"

    def test_round_down_str_never_exceeds_input(self):
        for value in (0.000001, 1.23456789, 999.999999):
            result = Decimal(round_down_str(value, 6))
            assert result <= Decimal(str(value))

    def test_to_decimal_truncates_to_8_places(self):
        assert to_decimal("1.123456789") == Decimal("1.12345678")

    def test_to_decimal_accepts_float_and_string_identically(self):
        assert to_decimal(0.1) == to_decimal("0.1")


class TestResolveGeneratorName:
    def test_resolves_real_function(self):
        fn = resolve_generator_name("common.utils:round_str")
        assert fn is round_str

    def test_returns_none_for_missing_colon(self):
        assert resolve_generator_name("not_a_module_path") is None

    def test_returns_none_for_unimportable_module(self):
        assert resolve_generator_name("no.such.module:fn") is None

    def test_returns_none_for_missing_function(self):
        assert resolve_generator_name("common.utils:no_such_function") is None


class TestFindAlgorithmByName:
    def test_finds_matching_entry(self):
        algorithms = [{"name": "a", "algo": "lc"}, {"name": "b", "algo": "gb"}]
        assert find_algorithm_by_name(algorithms, "b") == {"name": "b", "algo": "gb"}

    def test_raises_for_missing_name(self):
        with pytest.raises(StopIteration):
            find_algorithm_by_name([{"name": "a"}], "missing")


class TestMergeDataSources:
    def test_merges_two_sources_with_prefix(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        df1 = pd.DataFrame({"timestamp": idx, "close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        df2 = pd.DataFrame({"timestamp": idx, "close": [10.0, 20.0, 30.0, 40.0, 50.0]})

        data_sources = [
            {"folder": "A", "column_prefix": "", "df": df1},
            {"folder": "B", "column_prefix": "b", "df": df2},
        ]
        merged = merge_data_sources(data_sources, "timestamp", "1h", merge_interpolate=False)

        assert "close" in merged.columns
        assert "b_close" in merged.columns
        assert list(merged["close"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert list(merged["b_close"]) == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_interpolates_gaps_when_requested(self):
        idx_full = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        # Source with a gap in the middle (missing row at index 2)
        idx_sparse = idx_full.delete(2)
        df1 = pd.DataFrame({"timestamp": idx_full, "close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        df2 = pd.DataFrame({"timestamp": idx_sparse, "close": [10.0, 20.0, 40.0, 50.0]})

        data_sources = [
            {"folder": "A", "column_prefix": "", "df": df1},
            {"folder": "B", "column_prefix": "b", "df": df2},
        ]
        merged = merge_data_sources(data_sources, "timestamp", "1h", merge_interpolate=True)
        # The missing row should be linearly interpolated between 20 and 40 -> 30
        assert merged["b_close"].iloc[2] == pytest.approx(30.0)


class TestAppendDfDropConcat:
    def test_overlap_rows_from_new_df_win(self):
        idx1 = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        idx2 = pd.date_range("2026-01-01 03:00", periods=3, freq="h", tz="UTC")  # overlaps last 2 rows of idx1
        df1 = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx1)
        df2 = pd.DataFrame({"close": [999.0, 999.0, 999.0]}, index=idx2)

        result = append_df_drop_concat(df1, df2)

        assert len(result) == 6  # 3 non-overlapping old rows + 3 new rows
        assert result["close"].iloc[-1] == 999.0
        assert result.index.is_monotonic_increasing

    def test_empty_new_df_returns_original(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        df1 = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        empty = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([]))

        result = append_df_drop_concat(df1, empty)
        assert len(result) == 3
