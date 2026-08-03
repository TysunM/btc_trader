"""Config loading: JSONC config files + ``${VAR}`` secret placeholders resolved from .env/environment.

This is a clean split-out of what upstream ITB inlines into ``service/App.py``. Every script and
the online service both go through :func:`load_config` so there is exactly one place that knows
how to read a config file, keeping the documented offline/online parity invariant intact.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import json5
from dotenv import load_dotenv

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Sane defaults matching upstream ITB's App.config baseline file-name conventions.
# User config files only need to override what differs from these.
DEFAULT_CONFIG: dict[str, Any] = {
    "venue": "binance",
    "train": True,
    "time_column": "timestamp",
    "merge_file_name": "data.csv",
    "feature_file_name": "features.csv",
    "matrix_file_name": "matrix.csv",
    "predict_file_name": "predictions.csv",
    "signal_file_name": "signals.csv",
    "signal_models_file_name": "signal_models",
    "model_folder": "MODELS",
    "data_sources": [],
    "feature_sets": [],
    "label_sets": [],
    "train_feature_sets": [],
    "train_features": [],
    "labels": [],
    "algorithms": [],
    "signal_sets": [],
    "output_sets": [],
    "trade_model": {
        "enabled": False,
        "test_order_before_submit": True,
        # Deviation from upstream (which defaults this False, i.e. risky-by-default):
        # this port defaults to always-simulated unless a human explicitly flips it.
        "simulate_order_execution": True,
        "no_trades_only_data_processing": True,
        "percentage_used_for_trade": 99.0,
        "limit_price_adjustment": 0.005,
    },
}


class ConfigError(Exception):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto a copy of ``base``. Lists are replaced, not merged."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_placeholders(value: Any) -> Any:
    """Recursively replace ``${VAR_NAME}`` in strings with values from the environment.

    Missing variables resolve to an empty string rather than raising — most placeholders
    (API keys, telegram tokens) are legitimately unset for signals-only / offline usage.
    """
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            return os.environ.get(m.group(1), "")

        return _PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v) for v in value]
    return value


def load_config(config_file: str | Path, env_file: str | Path | None = None) -> dict[str, Any]:
    """Load a JSONC config file, merge onto :data:`DEFAULT_CONFIG`, resolve ``${VAR}`` secrets.

    :param config_file: path to a .jsonc/.json config file (comments/trailing commas allowed).
    :param env_file: optional explicit path to a .env file; defaults to python-dotenv's normal
        upward search from the current working directory.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    load_dotenv(dotenv_path=env_file, override=False)

    with config_path.open("r", encoding="utf-8") as f:
        try:
            raw = json5.load(f)
        except ValueError as e:
            raise ConfigError(f"Failed to parse {config_path}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level config in {config_path} must be an object")

    merged = _deep_merge(DEFAULT_CONFIG, raw)
    resolved = _resolve_placeholders(merged)
    resolved["config_file"] = str(config_path)
    return resolved


def require_fields(config: dict[str, Any], fields: list[str]) -> None:
    """Raise ConfigError listing every missing required field (not just the first)."""
    missing = [f for f in fields if not config.get(f)]
    if missing:
        raise ConfigError(f"Config {config.get('config_file', '?')} missing required fields: {missing}")
