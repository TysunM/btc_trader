"""Secondary backtest report: Sharpe ratio, max drawdown, trade list, and an equity-curve plot,
generated from an already-computed ``signals.csv`` via `backtesting.py`.

Usage: ``python -m reporting.run_report -c configs/btcusdt-1h.jsonc``

This is *additive* reporting, not a replacement for ``scripts/simulate.py``'s walk-forward grid
search — see ``reporting/bt_adapter.py``'s docstring for why ITB's own backtest stays the source
of truth for correctness.
"""

from __future__ import annotations

from pathlib import Path

import click

from common.config import load_config, require_fields
from common.io import symbol_data_path
from reporting.bt_adapter import run_backtest


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
@click.option("--cash", type=float, default=100_000.0, help="Starting cash for the backtest (see bt_adapter.run_backtest's docstring on why the default is higher than backtesting.py's own)")
@click.option("--commission", type=float, default=0.0, help="Per-side commission rate, e.g. 0.001 for 0.1%")
@click.option("--buy-signal-column", default="buy_signal_column", help="Boolean buy signal column name in signals.csv")
@click.option("--sell-signal-column", default="sell_signal_column", help="Boolean sell signal column name in signals.csv")
@click.option("--out", "out_file", type=click.Path(), default=None, help="Output HTML path (default: <data_folder>/<symbol>/bt_report.html)")
def main(config_file: str, cash: float, commission: float, buy_signal_column: str, sell_signal_column: str, out_file: str | None) -> None:
    config = load_config(config_file)
    require_fields(config, ["symbol", "data_folder", "time_column", "signal_file_name"])

    print(f"Loading signals and running backtest (cash={cash}, commission={commission})...")
    stats, bt = run_backtest(
        config,
        buy_signal_column=buy_signal_column,
        sell_signal_column=sell_signal_column,
        cash=cash,
        commission=commission,
    )

    print()
    print(stats)
    print()

    trades = stats.get("_trades")
    if trades is not None and len(trades):
        print(f"{len(trades)} trade(s). Win rate: {stats.get('Win Rate [%]'):.1f}%")
    else:
        print("No trades were executed with this signal/threshold combination.")

    out_path = Path(out_file) if out_file else symbol_data_path(config) / "bt_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        bt.plot(filename=str(out_path), open_browser=False)
        print(f"\nReport plot saved to: {out_path}")
    except Exception as e:
        print(f"\nWARNING: could not generate the HTML plot ({e}). Stats above are still valid.")


if __name__ == "__main__":
    main()
