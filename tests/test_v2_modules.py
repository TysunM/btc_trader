"""Tests for v2/regime_filter.py, v2/position_sizing.py, and v2/calibration.py's edge cases not
already covered via their integration tests in test_generators.py / test_backtesting.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.model_store import ModelStore
from v2.calibration import apply_calibrator, default_method_for_algo, fit_calibrator
from v2.position_sizing import atr_position_size, check_stop_take
from v2.regime_filter import generate_regime_gate


def _trending_ohlc_df(n=60) -> pd.DataFrame:
    """A strong, steady uptrend -- should register as a clear trending regime (high ADX)."""
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100, 200, n)
    return pd.DataFrame({"close": close, "high": close + 1, "low": close - 1}, index=idx)


def _choppy_ohlc_df(n=60) -> pd.DataFrame:
    """A flat, oscillating series with no directional trend -- should register as low ADX."""
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = 100 + 2 * np.sin(np.linspace(0, 20 * np.pi, n))
    return pd.DataFrame({"close": close, "high": close + 0.5, "low": close - 0.5}, index=idx)


class TestRegimeFilter:
    def test_trending_market_has_higher_regime_ok_rate_than_choppy(self, base_config):
        model_store = ModelStore(base_config)
        config = {"adx_window": 10, "adx_threshold": 20.0, "buy_signal_column": "buy_signal_column", "sell_signal_column": "sell_signal_column"}

        trend_df = _trending_ohlc_df()
        trend_df["buy_signal_column"] = True
        trend_df["sell_signal_column"] = False
        trend_result, _ = generate_regime_gate(trend_df.copy(), config, base_config, model_store)

        chop_df = _choppy_ohlc_df()
        chop_df["buy_signal_column"] = True
        chop_df["sell_signal_column"] = False
        chop_result, _ = generate_regime_gate(chop_df.copy(), config, base_config, model_store)

        assert trend_result["regime_ok"].mean() > chop_result["regime_ok"].mean()

    def test_gated_buy_signal_is_subset_of_original(self, base_config):
        model_store = ModelStore(base_config)
        config = {"adx_threshold": 20.0}
        df = _choppy_ohlc_df()
        df["buy_signal_column"] = True
        df["sell_signal_column"] = True
        result, _ = generate_regime_gate(df.copy(), config, base_config, model_store)

        # Gated buy signal can only be True where the original was True AND regime_ok is True
        assert (result["buy_signal_column"] <= df["buy_signal_column"]).all()

    def test_sell_signal_untouched_by_default(self, base_config):
        model_store = ModelStore(base_config)
        config = {"adx_threshold": 20.0}
        df = _choppy_ohlc_df()
        df["buy_signal_column"] = True
        df["sell_signal_column"] = True
        result, _ = generate_regime_gate(df.copy(), config, base_config, model_store)
        assert (result["sell_signal_column"] == True).all()  # noqa: E712

    def test_gate_sell_true_also_gates_exits(self, base_config):
        model_store = ModelStore(base_config)
        config = {"adx_threshold": 20.0, "gate_sell": True}
        df = _choppy_ohlc_df()
        df["buy_signal_column"] = True
        df["sell_signal_column"] = True
        result, features = generate_regime_gate(df.copy(), config, base_config, model_store)
        assert "sell_signal_column" in features
        assert not (result["sell_signal_column"] == True).all()  # noqa: E712 -- some gated off

    def test_missing_required_column_raises_clear_error(self, base_config):
        model_store = ModelStore(base_config)
        df = pd.DataFrame({"close": [1.0, 2.0]})  # missing high/low/buy/sell columns
        with pytest.raises(ValueError, match="not found"):
            generate_regime_gate(df, {}, base_config, model_store)


class TestPositionSizing:
    def test_zero_atr_returns_zero_fraction(self):
        assert atr_position_size(equity=10_000, atr=0, price=100) == 0.0

    def test_zero_price_returns_zero_fraction(self):
        assert atr_position_size(equity=10_000, atr=5, price=0) == 0.0

    def test_zero_equity_returns_zero_fraction(self):
        assert atr_position_size(equity=0, atr=5, price=100) == 0.0

    def test_larger_atr_reduces_position_size(self):
        # Chosen so neither hits kelly_cap (0.25 default) -- fraction = risk_pct% * price / (atr * 100)
        small_atr_size = atr_position_size(equity=10_000, atr=3000, price=50_000, risk_pct=1.0)
        large_atr_size = atr_position_size(equity=10_000, atr=6000, price=50_000, risk_pct=1.0)
        assert 0 < large_atr_size < 0.25
        assert 0 < small_atr_size < 0.25
        assert large_atr_size < small_atr_size

    def test_result_never_exceeds_kelly_cap(self):
        # Tiny ATR would imply a huge (unrealistic) position without the cap.
        fraction = atr_position_size(equity=10_000, atr=0.0001, price=100, risk_pct=1.0, kelly_cap=0.25)
        assert fraction == 0.25

    def test_result_is_never_negative(self):
        fraction = atr_position_size(equity=10_000, atr=50, price=50_000, risk_pct=1.0)
        assert fraction >= 0.0


class TestStopLossTakeProfit:
    def test_no_thresholds_configured_never_triggers(self):
        assert check_stop_take(100.0, 50.0, None, None) is None

    def test_stop_loss_triggers_on_sufficient_adverse_move(self):
        assert check_stop_take(100.0, 97.0, stop_loss_pct=2.0, take_profit_pct=None) == "stop_loss"

    def test_stop_loss_does_not_trigger_below_threshold(self):
        assert check_stop_take(100.0, 99.0, stop_loss_pct=2.0, take_profit_pct=None) is None

    def test_take_profit_triggers_on_sufficient_favorable_move(self):
        assert check_stop_take(100.0, 106.0, stop_loss_pct=None, take_profit_pct=5.0) == "take_profit"

    def test_take_profit_does_not_trigger_below_threshold(self):
        assert check_stop_take(100.0, 103.0, stop_loss_pct=None, take_profit_pct=5.0) is None

    def test_invalid_entry_price_never_triggers(self):
        assert check_stop_take(0.0, 50.0, stop_loss_pct=1.0, take_profit_pct=1.0) is None

    def test_exact_boundary_counts_as_triggered(self):
        assert check_stop_take(100.0, 95.0, stop_loss_pct=5.0, take_profit_pct=None) == "stop_loss"


class TestCalibration:
    def test_isotonic_calibrator_output_bounded_zero_one(self):
        rng = np.random.default_rng(0)
        scores = pd.Series(rng.normal(0, 3, 200))
        labels = pd.Series((scores > 0).astype(int))
        calibrator = fit_calibrator(scores, labels, method="isotonic")
        calibrated = apply_calibrator(calibrator, scores)
        assert calibrated.between(0, 1).all()

    def test_sigmoid_calibrator_output_bounded_zero_one(self):
        rng = np.random.default_rng(1)
        scores = pd.Series(rng.normal(0, 3, 200))
        labels = pd.Series((scores > 0).astype(int))
        calibrator = fit_calibrator(scores, labels, method="sigmoid")
        calibrated = apply_calibrator(calibrator, scores)
        assert calibrated.between(0, 1).all()

    def test_nan_positions_preserved_through_calibration(self):
        rng = np.random.default_rng(2)
        scores = pd.Series(rng.normal(0, 3, 200))
        labels = pd.Series((scores > 0).astype(int))
        calibrator = fit_calibrator(scores, labels, method="isotonic")

        test_scores = pd.Series([0.5, np.nan, -0.5, np.nan, 1.0])
        calibrated = apply_calibrator(calibrator, test_scores)
        assert calibrated.isna().equals(test_scores.isna())

    def test_raises_on_too_few_samples(self):
        scores = pd.Series([0.1, 0.2, 0.3])
        labels = pd.Series([0, 1, 1])
        with pytest.raises(ValueError, match="at least 10"):
            fit_calibrator(scores, labels)

    def test_raises_on_unknown_method(self):
        scores = pd.Series(range(20), dtype=float)
        labels = pd.Series([0, 1] * 10)
        with pytest.raises(ValueError, match="Unknown calibration method"):
            fit_calibrator(scores, labels, method="not_a_real_method")

    def test_default_method_for_algo(self):
        assert default_method_for_algo("gb") == "isotonic"
        assert default_method_for_algo("nn") == "isotonic"
        assert default_method_for_algo("lc") == "sigmoid"
        assert default_method_for_algo("svc") == "sigmoid"

    def test_all_nan_input_returns_copy_without_error(self):
        rng = np.random.default_rng(3)
        scores = pd.Series(rng.normal(0, 3, 200))
        labels = pd.Series((scores > 0).astype(int))
        calibrator = fit_calibrator(scores, labels, method="isotonic")

        all_nan = pd.Series([np.nan, np.nan])
        result = apply_calibrator(calibrator, all_nan)
        assert result.isna().all()
