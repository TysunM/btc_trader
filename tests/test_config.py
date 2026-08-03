"""Tests for common/config.py: JSONC parsing (comments/trailing commas), .env-style secret
placeholder resolution, default-merging, and required-field validation.
"""

from __future__ import annotations

import pytest

from common.config import ConfigError, load_config, require_fields


@pytest.fixture
def write_config(tmp_path):
    def _write(content: str, name: str = "test.jsonc") -> str:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    return _write


class TestJsoncParsing:
    def test_parses_line_comments(self, write_config):
        path = write_config('{\n  // this is a comment\n  "symbol": "BTCUSDT"\n}')
        config = load_config(path)
        assert config["symbol"] == "BTCUSDT"

    def test_parses_trailing_commas(self, write_config):
        path = write_config('{\n  "symbol": "BTCUSDT",\n  "freq": "1h",\n}')
        config = load_config(path)
        assert config["freq"] == "1h"

    def test_raises_config_error_on_invalid_json(self, write_config):
        path = write_config('{\n  "symbol": "BTCUSDT"\n  "freq": "1h"\n}')  # missing comma
        with pytest.raises(ConfigError):
            load_config(path)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(tmp_path / "does_not_exist.jsonc"))

    def test_raises_on_non_object_top_level(self, write_config):
        path = write_config("[1, 2, 3]")
        with pytest.raises(ConfigError, match="object"):
            load_config(path)


class TestSecretPlaceholderResolution:
    def test_resolves_env_var_placeholder(self, write_config, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_TOKEN", "abc123")
        path = write_config('{"telegram_bot_token": "${TEST_SECRET_TOKEN}"}')
        config = load_config(path)
        assert config["telegram_bot_token"] == "abc123"

    def test_missing_env_var_resolves_to_empty_string_not_error(self, write_config, monkeypatch):
        monkeypatch.delenv("TEST_UNSET_VAR_XYZ", raising=False)
        path = write_config('{"api_key": "${TEST_UNSET_VAR_XYZ}"}')
        config = load_config(path)
        assert config["api_key"] == ""

    def test_resolves_placeholders_nested_in_dict(self, write_config, monkeypatch):
        monkeypatch.setenv("TEST_NESTED_VAL", "nested-value")
        path = write_config('{"trade_model": {"note": "${TEST_NESTED_VAL}"}}')
        config = load_config(path)
        assert config["trade_model"]["note"] == "nested-value"

    def test_resolves_placeholders_nested_in_list(self, write_config, monkeypatch):
        monkeypatch.setenv("TEST_LIST_VAL", "list-value")
        path = write_config('{"items": ["${TEST_LIST_VAL}", "literal"]}')
        config = load_config(path)
        assert config["items"] == ["list-value", "literal"]

    def test_literal_string_without_placeholder_syntax_untouched(self, write_config):
        path = write_config('{"symbol": "BTCUSDT"}')
        config = load_config(path)
        assert config["symbol"] == "BTCUSDT"

    def test_never_reads_real_credentials_by_accident(self, write_config, monkeypatch):
        # Simulates the real .env scenario: an unset secret must never silently pick up some
        # unrelated environment value -- it must resolve to empty, not raise or guess.
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        path = write_config('{"api_key": "${BINANCE_API_KEY}"}')
        config = load_config(path)
        assert config["api_key"] == ""


class TestDefaultMerging:
    def test_user_config_overrides_defaults(self, write_config):
        path = write_config('{"model_folder": "CUSTOM_MODELS"}')
        config = load_config(path)
        assert config["model_folder"] == "CUSTOM_MODELS"

    def test_unspecified_fields_get_default_values(self, write_config):
        path = write_config('{"symbol": "BTCUSDT"}')
        config = load_config(path)
        assert config["merge_file_name"] == "data.csv"
        assert config["model_folder"] == "MODELS"

    def test_trade_model_defaults_are_safe(self, write_config):
        # The core safety property from common/config.py's DEFAULT_CONFIG: simulate_order_execution
        # must default true (flipped from upstream's risky-by-default false).
        path = write_config('{"symbol": "BTCUSDT"}')
        config = load_config(path)
        assert config["trade_model"]["simulate_order_execution"] is True
        assert config["trade_model"]["enabled"] is False

    def test_partial_trade_model_override_merges_not_replaces(self, write_config):
        path = write_config('{"trade_model": {"stop_loss_pct": 3.0}}')
        config = load_config(path)
        assert config["trade_model"]["stop_loss_pct"] == 3.0
        assert config["trade_model"]["simulate_order_execution"] is True  # default preserved

    def test_config_file_path_recorded(self, write_config):
        path = write_config('{"symbol": "BTCUSDT"}')
        config = load_config(path)
        assert config["config_file"] == path


class TestRequireFields:
    def test_passes_when_all_fields_present(self):
        require_fields({"a": 1, "b": 2}, ["a", "b"])  # should not raise

    def test_lists_all_missing_fields_not_just_first(self):
        with pytest.raises(ConfigError) as exc_info:
            require_fields({"a": 1}, ["a", "b", "c"])
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)

    def test_falsy_but_present_values_count_as_missing(self):
        # Empty list/0/"" for a required field should be flagged, not silently accepted --
        # e.g. an empty data_sources list is not usable even though the key technically exists.
        with pytest.raises(ConfigError):
            require_fields({"data_sources": []}, ["data_sources"])
