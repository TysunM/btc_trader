"""Tests for common/backtesting.py, including regression tests for the v2 fee/slippage model
(CHANGELOG_V2.md item 1) — this pass found and fixed a real bug here (fee cancelling out
between legs instead of compounding), so these tests exist specifically to catch a regression
of that bug, not just to check the feature exists.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.backtesting import simulated_trade_performance
from v2.fees import round_trip_cost_pct, sharpe_like_ratio


def _signals_df(prices: list[float], buys: list[bool], sells: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"sell_signal_column": sells, "buy_signal_column": buys, "close": prices})


class TestSimulatedTradePerformanceBaseline:
    """v1 baseline behavior — must stay exact (fee_bps=0, slippage_bps=0 is the default)."""

    def test_no_signals_no_transactions(self):
        df = _signals_df([100, 101, 102], [False, False, False], [False, False, False])
        performance, long_perf, short_perf = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close")
        assert performance["#transactions"] == 0
        assert performance["profit"] == 0

    def test_first_transaction_in_each_bucket_has_zero_profit(self):
        # The very first buy/sell in a bucket has no prior reference price (previous_price=0),
        # so profit is 0 by design -- this is upstream's own semantics, not a bug.
        df = _signals_df([100, 105, 103, 110], [True, False, False, False], [False, False, False, True])
        _, long_perf, short_perf = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close")
        assert short_perf["#transactions"] == 1
        assert short_perf["profit"] == 0.0
        assert long_perf["#transactions"] == 1
        assert long_perf["profit"] == 0.0

    def test_alternating_signals_compute_expected_deltas(self):
        # buy@100 (init, profit 0) -> sell@110 (init long, profit 0) -> buy@105 (short delta
        # 100-105=-5) -> sell@115 (long delta 115-110=+5)
        prices = [100, 110, 105, 115]
        buys = [True, False, True, False]
        sells = [False, True, False, True]
        df = _signals_df(prices, buys, sells)
        _, long_perf, short_perf = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close")

        assert short_perf["#transactions"] == 2
        assert short_perf["profit"] == pytest.approx(0.0 + (100 - 105))
        assert long_perf["#transactions"] == 2
        assert long_perf["profit"] == pytest.approx(0.0 + (115 - 110))

    def test_null_price_rows_are_skipped(self):
        df = pd.DataFrame(
            {"sell_signal_column": [False, False], "buy_signal_column": [True, True], "close": [100.0, float("nan")]}
        )
        performance, _, _ = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close")
        assert performance["#transactions"] == 1  # the NaN-price row must not count


class TestFeeSlippageRegression:
    """Regression coverage for the bug found during v2 verification: fee/slippage must
    compound across repeated same-side transactions, not cancel out."""

    def test_zero_fee_matches_baseline_exactly(self):
        prices = [100, 110, 90, 120, 80, 130]
        buys = [True, False, True, False, True, False]
        sells = [False, True, False, True, False, True]
        df = _signals_df(prices, buys, sells)

        perf_baseline, long_baseline, _ = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close")
        perf_zero_fee, long_zero_fee, _ = simulated_trade_performance(
            df, "buy_signal_column", "sell_signal_column", "close", fee_bps=0.0, slippage_bps=0.0
        )
        assert long_baseline == long_zero_fee

    def test_fee_reduces_every_completed_trades_return_by_exact_round_trip_cost(self):
        # Three long round trips with real (non-zero previous_price) deltas.
        prices = [100, 110, 108, 120, 90, 130]
        buys = [True, False, True, False, True, False]
        sells = [False, True, False, True, False, True]
        df = _signals_df(prices, buys, sells)

        fee_bps, slippage_bps = 10.0, 2.0
        expected_cost_pct = round_trip_cost_pct(fee_bps, slippage_bps)
        assert expected_cost_pct == pytest.approx(0.24)  # 2 * (10+2)/100

        _, long_zero, _ = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close")
        _, long_fee, _ = simulated_trade_performance(
            df, "buy_signal_column", "sell_signal_column", "close", fee_bps=fee_bps, slippage_bps=slippage_bps
        )

        zero_returns = long_zero["_trade_returns_pct"]
        fee_returns = long_fee["_trade_returns_pct"]
        assert len(zero_returns) == len(fee_returns) > 0
        for z, f in zip(zero_returns, fee_returns):
            assert f == pytest.approx(z - expected_cost_pct, abs=1e-9)

    def test_fee_never_cancels_out_across_repeated_transactions(self):
        # This is the specific regression this test file exists to catch: with the original
        # (buggy) per-leg price-adjustment approach, fees on consecutive same-side legs mostly
        # cancelled out. With N completed round trips, total profit reduction must scale
        # linearly with N, not stay roughly constant.
        prices = [100, 110, 105, 115, 108, 120, 112, 125]
        buys = [True, False, True, False, True, False, True, False]
        sells = [False, True, False, True, False, True, False, True]
        df = _signals_df(prices, buys, sells)

        _, long_zero, _ = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close")
        _, long_fee, _ = simulated_trade_performance(
            df, "buy_signal_column", "sell_signal_column", "close", fee_bps=10.0, slippage_bps=2.0
        )

        n_trades = long_zero["#transactions"]
        assert n_trades >= 3
        total_cost = long_zero["profit"] - long_fee["profit"]
        cost_per_trade = total_cost / n_trades
        # Each trade's price is ~100-125, 0.24% of that is roughly 0.25-0.30 per trade; the
        # bug this guards against would produce a cost_per_trade near 0 (cancellation).
        assert cost_per_trade > 0.15

    def test_higher_fee_never_improves_profit(self):
        prices = [100, 110, 105, 118, 95, 130]
        buys = [True, False, True, False, True, False]
        sells = [False, True, False, True, False, True]
        df = _signals_df(prices, buys, sells)

        _, long_low_fee, _ = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close", fee_bps=1.0, slippage_bps=0.0)
        _, long_high_fee, _ = simulated_trade_performance(df, "buy_signal_column", "sell_signal_column", "close", fee_bps=50.0, slippage_bps=10.0)
        assert long_high_fee["profit"] <= long_low_fee["profit"]


class TestSharpeLikeRatio:
    def test_empty_list_is_zero(self):
        assert sharpe_like_ratio([]) == 0.0

    def test_single_trade_is_zero(self):
        assert sharpe_like_ratio([5.0]) == 0.0

    def test_zero_variance_is_zero(self):
        assert sharpe_like_ratio([2.0, 2.0, 2.0]) == 0.0

    def test_positive_consistent_returns_give_positive_ratio(self):
        assert sharpe_like_ratio([1.0, 1.2, 0.9, 1.1]) > 0

    def test_higher_volatility_lowers_ratio_for_same_mean(self):
        consistent = sharpe_like_ratio([1.0, 1.1, 0.9, 1.0])
        volatile = sharpe_like_ratio([3.0, -1.0, 3.0, -1.0])
        assert consistent > volatile
