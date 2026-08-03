"""End-to-end sanity harness: runs the full offline pipeline against real market data and
asserts a set of sanity conditions on the outputs. This is the "did I actually break the
pipeline" check that unit tests alone can't give — meant to be run manually (or on a schedule)
after any change that touches the pipeline, not as part of the fast unit-test suite.

Usage: python -m scripts.smoke_test -c configs/btcusdt-1h.jsonc
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from common.config import load_config, require_fields
from common.io import read_data_file, symbol_data_path


def _run_stage(label: str, callback, config_file: str) -> None:
    print(f"\n--- Running {label} ---")
    callback(config_file=config_file)


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "data_folder", "time_column", "freq", "label_horizon"])

    now = datetime.now()
    print(f"=== Smoke test: {config_file} ===")

    # Import here (not at module top level) so a syntax/import error in one pipeline stage is
    # attributed to that stage clearly, rather than failing this whole script's own import.
    from scripts import download, features, labels, merge, predict_rolling, signals, simulate, train

    for label, module in [
        ("download", download),
        ("merge", merge),
        ("features", features),
        ("labels", labels),
        ("train", train),
        ("predict_rolling", predict_rolling),
        ("signals", signals),
        ("simulate", simulate),
    ]:
        _run_stage(label, module.main.callback, config_file)

    print("\n=== Sanity checks ===")
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, condition))
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    time_column = config["time_column"]
    data_path = symbol_data_path(config)

    data_df = read_data_file(data_path / config["merge_file_name"], time_column)
    check("data.csv is non-empty", len(data_df) > 0)

    freq_td = pd.Timedelta(config["freq"])
    diffs = data_df[time_column].diff().dropna().unique()
    check("data.csv has a single, continuous timestamp frequency", len(diffs) == 1 and pd.Timedelta(diffs[0]) == freq_td)

    matrix_df = read_data_file(data_path / config["matrix_file_name"], time_column)
    label_horizon = config["label_horizon"]
    label_cols = [c for c in config.get("labels", []) if c in matrix_df.columns]
    if label_cols:
        non_tail = matrix_df.iloc[:-label_horizon] if label_horizon else matrix_df
        check("matrix.csv labels have no unexpected NaN outside the label_horizon tail", not non_tail[label_cols].isna().any().any())

    model_path = Path(config["data_folder"]) / config["symbol"] / config["model_folder"]
    model_files = list(model_path.glob("*.pickle"))
    check("at least one model file was written", len(model_files) > 0)
    check("no model file is empty (0 bytes)", all(f.stat().st_size > 0 for f in model_files))

    signals_df = read_data_file(data_path / config["signal_file_name"], time_column)
    if "buy_signal_column" in signals_df.columns:
        check("signals.csv has at least one buy signal", bool(signals_df["buy_signal_column"].any()))
    if "sell_signal_column" in signals_df.columns:
        check("signals.csv has at least one sell signal", bool(signals_df["sell_signal_column"].any()))

    signal_models_path = (data_path / config["signal_models_file_name"]).with_suffix(".txt")
    check("signal_models.txt (simulate.py output) exists", signal_models_path.is_file())

    n_passed = sum(1 for _, ok in checks if ok)
    print(f"\n{n_passed}/{len(checks)} checks passed. Elapsed: {str(datetime.now() - now).split('.')[0]}")

    failed = [name for name, ok in checks if not ok]
    if failed:
        raise SystemExit(f"Smoke test FAILED: {failed}")


if __name__ == "__main__":
    main()
