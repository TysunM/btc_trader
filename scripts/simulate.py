"""Grid-search over signal thresholds, backtesting each combination with
:func:`common.backtesting.simulated_trade_performance` and keeping the top-N by monthly profit.

Safety note vs. upstream: upstream evaluates grid string values with raw ``eval()``. Since a
config file's grid values are always literal lists (e.g. ``"[0.01, 0.02, ..., 0.10]"``, never an
expression needing real code execution), this port uses ``ast.literal_eval`` instead — same
behavior for every value ITB's own sample configs actually use, without executing arbitrary code
from a config file.

v2 additions (CHANGELOG_V2.md items 1 and 7), both opt-in via ``simulate_model``:
``fee_bps``/``slippage_bps`` (default 0/0, reproducing the v1 fee-less baseline) deduct realistic
Binance execution costs from every simulated transaction, and ``rank_by: "sharpe"`` (default
``"%profit/M"``, matching v1) ranks the threshold grid by a per-trade-return-volatility-adjusted
score instead of raw monthly profit, penalizing configs whose profit came from a few lucky
high-variance trades.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta

import click
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from common.backtesting import simulated_trade_performance
from common.config import load_config, require_fields
from common.generators import generate_feature_set
from common.io import read_data_file, symbol_data_path
from common.model_store import ModelStore
from v2.fees import sharpe_like_ratio


def _maybe_literal_eval(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
def main(config_file: str) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "data_folder", "time_column", "signal_sets", "simulate_model"])

    model_store = ModelStore(config)
    model_store.load_models()

    time_column = config["time_column"]
    data_path = symbol_data_path(config)
    now = datetime.now()

    file_path = data_path / config["signal_file_name"]
    print(f"Loading signals from {file_path}...")
    df = read_data_file(file_path, time_column)
    print(f"Loaded {len(df)} records / {len(df.columns)} columns.")

    simulate_config = config["simulate_model"]
    data_start = simulate_config.get("data_start")
    data_end = simulate_config.get("data_end")
    if data_start:
        df = df[df[time_column] >= data_start] if isinstance(data_start, str) else df.iloc[data_start:]
    if data_end:
        df = df[df[time_column] < data_end] if isinstance(data_end, str) else df.iloc[:-data_end]
    df = df.reset_index(drop=True)

    print(f"Input data size {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    parameter_grid = dict(simulate_config.get("grid", {}))
    direction = simulate_config.get("direction", "")
    if direction not in ("long", "short"):
        raise ValueError(f"simulate_model.direction must be 'long' or 'short', got {direction!r}.")
    topn_to_store = simulate_config.get("topn_to_store", 10)

    # v2 items 1 and 7 (CHANGELOG_V2.md), both opt-in and 0/off by default (v1 baseline).
    fee_bps = simulate_config.get("fee_bps", 0.0)
    slippage_bps = simulate_config.get("slippage_bps", 0.0)
    rank_by = simulate_config.get("rank_by", "%profit/M")
    if rank_by not in ("%profit/M", "sharpe"):
        raise ValueError(f"simulate_model.rank_by must be '%profit/M' or 'sharpe', got {rank_by!r}.")

    for key in ("buy_signal_threshold", "buy_signal_threshold_2", "sell_signal_threshold", "sell_signal_threshold_2"):
        if key in parameter_grid:
            parameter_grid[key] = _maybe_literal_eval(parameter_grid[key])

    if simulate_config.get("buy_sell_equal"):
        parameter_grid["sell_signal_threshold"] = [None]
        parameter_grid["sell_signal_threshold_2"] = [None]

    months_in_simulation = (df[time_column].iloc[-1] - df[time_column].iloc[0]) / timedelta(days=365 / 12)

    generator_name = simulate_config.get("signal_generator")
    signal_generator = next((ss for ss in config.get("signal_sets", []) if ss.get("generator") == generator_name), None)
    if not signal_generator:
        raise ValueError(f"Signal generator {generator_name!r} not found in signal_sets.")

    performances = []
    for parameters in tqdm(ParameterGrid([parameter_grid]), desc="grid"):
        if simulate_config.get("buy_sell_equal"):
            parameters["sell_signal_threshold"] = -parameters["buy_signal_threshold"]
            if parameters.get("buy_signal_threshold_2") is not None:
                parameters["sell_signal_threshold_2"] = -parameters["buy_signal_threshold_2"]

        signal_generator["config"]["parameters"].update(parameters)
        df, _ = generate_feature_set(df, signal_generator, config, model_store, last_rows=0)

        buy_signal_column, sell_signal_column = signal_generator["config"]["names"]
        performance, long_performance, short_performance = simulated_trade_performance(
            df, buy_signal_column, sell_signal_column, "close", fee_bps=fee_bps, slippage_bps=slippage_bps
        )
        performance = long_performance if direction == "long" else short_performance

        performance["#transactions/M"] = round(performance["#transactions"] / months_in_simulation, 2)
        performance["profit/M"] = round(performance["profit"] / months_in_simulation, 2)
        performance["%profit/M"] = round(performance["%profit"] / months_in_simulation, 2)
        performance["sharpe"] = round(sharpe_like_ratio(performance.pop("_trade_returns_pct", [])), 3)

        performances.append({"model": dict(parameters), "performance": performance})

    performances.sort(key=lambda x: x["performance"][rank_by], reverse=True)
    performances = performances[:topn_to_store]

    if not performances:
        print("No parameter combinations produced results.")
        return

    keys = list(performances[0]["model"].keys()) + list(performances[0]["performance"].keys())
    lines = [
        ",".join(str(v) for v in (list(p["model"].values()) + list(p["performance"].values())))
        for p in performances
    ]

    out_path = (data_path / config["signal_models_file_name"]).with_suffix(".txt")
    add_header = not out_path.is_file()
    with open(out_path, "a+") as f:
        if add_header:
            f.write(",".join(keys) + "\n")
        f.write("\n".join(lines) + "\n\n")

    print(f"\nTop {len(performances)} result(s) by {rank_by}:")
    for p in performances[:5]:
        print(f"  {p['model']} -> {p['performance']}")

    print(f"\nSimulation results stored in: {out_path}")
    print(f"Finished simulation in {str(datetime.now() - now).split('.')[0]}")


if __name__ == "__main__":
    main()
