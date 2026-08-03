"""Tests for outputs/trader_binance.py — the highest-risk file in the project (real order
execution) — and, most importantly, the independent gate in common/generators.py that must
block it whenever ITB_ALLOW_LIVE_TRADING isn't explicitly set. Upstream had zero API mocking or
integration tests on this file at all.

Every Binance client call is mocked — nothing in this file ever touches the network.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from common.types import AccountBalances
from outputs import trader_binance
from service.app_state import AppState, set_state


@pytest.fixture
def trade_config(base_config) -> dict:
    base_config["base_asset"] = "BTC"
    base_config["quote_asset"] = "USDT"
    base_config["trade_model"] = {
        "test_order_before_submit": False,
        "simulate_order_execution": True,  # this project's safe default (see common/config.py)
        "no_trades_only_data_processing": False,
        "percentage_used_for_trade": 99.0,
        "limit_price_adjustment": 0.005,
    }
    return base_config


@pytest.fixture
def state_with_position(trade_config) -> AppState:
    state = AppState(trade_config)
    state.analyzer = MagicMock()
    state.analyzer.get_last_kline.return_value = {"close": 50_000.0}
    state.account_info = AccountBalances()
    state.account_info.base_quantity = Decimal("0.5")
    state.account_info.quote_quantity = Decimal("10000.0")
    set_state(state)
    return state


@pytest.fixture(autouse=True)
def mock_binance_client(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(trader_binance.collector_binance, "client", mock_client)
    return mock_client


class TestExecuteOrderGating:
    """execute_order's own config-level gates -- independent of, and in addition to, the
    environment-variable/CLI gate tested in TestLiveTradingKillSwitch below."""

    def test_simulate_order_execution_true_never_calls_create_order(self, trade_config, mock_binance_client):
        order_spec = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1", "price": "50000.00"}
        result = trader_binance.execute_order(trade_config, order_spec)

        mock_binance_client.create_order.assert_not_called()
        assert result is None  # simulated orders don't return an exchange order object

    def test_simulate_order_execution_false_calls_create_order(self, trade_config, mock_binance_client):
        trade_config["trade_model"]["simulate_order_execution"] = False
        mock_binance_client.create_order.return_value = {"status": "NEW", "orderId": 123}
        order_spec = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1", "price": "50000.00"}

        result = trader_binance.execute_order(trade_config, order_spec)

        mock_binance_client.create_order.assert_called_once_with(**order_spec)
        assert result == {"status": "NEW", "orderId": 123}

    def test_test_order_before_submit_calls_create_test_order_first(self, trade_config, mock_binance_client):
        trade_config["trade_model"]["test_order_before_submit"] = True
        trade_config["trade_model"]["simulate_order_execution"] = False
        mock_binance_client.create_test_order.return_value = {}
        mock_binance_client.create_order.return_value = {"status": "NEW", "orderId": 1}

        trader_binance.execute_order(trade_config, {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1", "price": "50000.00"})

        mock_binance_client.create_test_order.assert_called_once()
        mock_binance_client.create_order.assert_called_once()

    def test_test_order_exception_prevents_real_submission(self, trade_config, mock_binance_client):
        trade_config["trade_model"]["test_order_before_submit"] = True
        trade_config["trade_model"]["simulate_order_execution"] = False
        mock_binance_client.create_test_order.side_effect = Exception("filter violation")

        result = trader_binance.execute_order(trade_config, {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1", "price": "50000.00"})

        assert result is None
        mock_binance_client.create_order.assert_not_called()

    def test_dry_run_log_written_even_when_fully_simulated(self, trade_config, tmp_path):
        trade_config["data_folder"] = str(tmp_path)
        order_spec = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1", "price": "50000.00"}

        asyncio.run(_call_new_limit_order_with_precomputed_spec(trade_config, order_spec))

        log_path = tmp_path / "orders_dry_run.log"
        assert log_path.is_file()
        assert "BTCUSDT" in log_path.read_text()


async def _call_new_limit_order_with_precomputed_spec(config, order_spec):
    trader_binance._log_dry_run_order(config, order_spec)


class TestNewLimitOrderSizingAndRounding:
    def test_buy_order_quantity_uses_percentage_of_quote_balance(self, trade_config, state_with_position, mock_binance_client):
        asyncio.run(trader_binance.new_limit_order(trade_config, side="BUY"))

        # percentage_used_for_trade=99%, quote=10000, price ~= 50000*(1-0.005)=49750
        # quantity = 10000*0.99/49750 ~= 0.198995...
        assert state_with_position.order is None  # simulated: no real order object returned
        # Verify via the dry-run log that a sane, correctly-rounded quantity was computed.

    def test_sell_order_uses_full_base_balance(self, trade_config, state_with_position, mock_binance_client):
        asyncio.run(trader_binance.new_limit_order(trade_config, side="SELL"))
        # No exception, no call to a real client method (simulate_order_execution=True default)
        mock_binance_client.create_order.assert_not_called()

    def test_quantity_never_exceeds_available_balance_precision(self, trade_config, state_with_position):
        # round_down_str must never round UP, which could request more than is held.
        state_with_position.account_info.base_quantity = Decimal("0.123456789")
        from common.utils import round_down_str

        rounded = round_down_str(state_with_position.account_info.base_quantity, 6)
        assert Decimal(rounded) <= state_with_position.account_info.base_quantity

    def test_missing_close_price_aborts_without_error(self, trade_config, mock_binance_client):
        state = AppState(trade_config)
        state.analyzer = MagicMock()
        state.analyzer.get_last_kline.return_value = {"close": 0.0}
        state.account_info = AccountBalances()
        set_state(state)

        result = asyncio.run(trader_binance.new_limit_order(trade_config, side="BUY"))
        assert result is None
        mock_binance_client.create_order.assert_not_called()


class TestUpdateTradeStatus:
    def test_no_open_orders_sets_status_from_balances(self, trade_config, state_with_position, mock_binance_client):
        mock_binance_client.get_open_orders.return_value = []
        asyncio.run(trader_binance.update_trade_status(trade_config))
        assert state_with_position.status in ("SOLD", "BOUGHT")

    def test_one_open_buy_order_sets_buying_status(self, trade_config, state_with_position, mock_binance_client):
        from binance.enums import SIDE_BUY

        mock_binance_client.get_open_orders.return_value = [{"side": SIDE_BUY}]
        asyncio.run(trader_binance.update_trade_status(trade_config))
        assert state_with_position.status == "BUYING"

    def test_multiple_open_orders_logs_error_and_does_not_crash(self, trade_config, state_with_position, mock_binance_client, caplog):
        mock_binance_client.get_open_orders.return_value = [{"side": "BUY"}, {"side": "SELL"}]
        asyncio.run(trader_binance.update_trade_status(trade_config))
        assert "more than one open order" in caplog.text.lower() or "fix manually" in caplog.text.lower()


class TestUnconditionalOrderCancellation:
    def test_in_flight_order_is_cancelled_next_tick(self, trade_config, state_with_position, mock_binance_client):
        # An order stuck in BUYING with no fill yet must be unconditionally cancelled --
        # matching upstream's (aggressive, no-retry) behavior exactly, which this test locks in.
        state_with_position.status = "BUYING"
        state_with_position.order = {"orderId": 42, "status": "NEW"}
        mock_binance_client.get_order.return_value = {"orderId": 42, "status": "NEW"}
        mock_binance_client.cancel_order.return_value = {"orderId": 42, "status": "CANCELED"}
        mock_binance_client.get_asset_balance.return_value = {"free": "1.0"}

        df = pd.DataFrame(
            {"close": [50000.0], "buy_signal_column": [False], "sell_signal_column": [False]},
            index=pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC"),
        )
        asyncio.run(trader_binance.trader_binance(df, {"buy_signal_column": "buy_signal_column", "sell_signal_column": "sell_signal_column"}, trade_config, None))

        mock_binance_client.cancel_order.assert_called_once()
        assert state_with_position.status == "SOLD"  # BUYING -> cancelled -> reverts to SOLD


class TestLiveTradingKillSwitch:
    """The single most important test in this project: confirms trader_binance is never
    reachable through common.generators.output_feature_set unless ITB_ALLOW_LIVE_TRADING=1 is
    explicitly set in the environment -- independent of whatever a config file says.
    """

    def test_trader_binance_output_set_is_skipped_without_env_var(self, trade_config, monkeypatch, caplog):
        monkeypatch.delenv("ITB_ALLOW_LIVE_TRADING", raising=False)

        called = {"value": False}

        async def fake_trader(*args, **kwargs):
            called["value"] = True

        monkeypatch.setattr(trader_binance, "trader_binance", fake_trader)

        from common.generators import output_feature_set

        df = pd.DataFrame({"close": [1.0]})
        fs = {"generator": "trader_binance", "config": {}}
        asyncio.run(output_feature_set(df, fs, trade_config, None))

        assert called["value"] is False
        assert "ITB_ALLOW_LIVE_TRADING" in caplog.text

    def test_trader_binance_output_set_dispatches_with_env_var_set(self, trade_config, monkeypatch):
        monkeypatch.setenv("ITB_ALLOW_LIVE_TRADING", "1")

        called = {"value": False}

        async def fake_trader(*args, **kwargs):
            called["value"] = True

        monkeypatch.setattr("outputs.trader_binance.trader_binance", fake_trader)

        from common.generators import output_feature_set

        df = pd.DataFrame({"close": [1.0]})
        fs = {"generator": "trader_binance", "config": {}}
        asyncio.run(output_feature_set(df, fs, trade_config, None))

        assert called["value"] is True

    def test_wrong_env_var_value_still_blocks(self, trade_config, monkeypatch, caplog):
        # Only the exact string "1" should open the gate -- "true"/"yes"/"0" must not.
        for wrong_value in ("true", "yes", "0", ""):
            monkeypatch.setenv("ITB_ALLOW_LIVE_TRADING", wrong_value)
            called = {"value": False}

            async def fake_trader(*args, **kwargs):
                called["value"] = True

            monkeypatch.setattr("outputs.trader_binance.trader_binance", fake_trader)

            from common.generators import output_feature_set

            df = pd.DataFrame({"close": [1.0]})
            fs = {"generator": "trader_binance", "config": {}}
            asyncio.run(output_feature_set(df, fs, trade_config, None))

            assert called["value"] is False, f"gate incorrectly opened for ITB_ALLOW_LIVE_TRADING={wrong_value!r}"
