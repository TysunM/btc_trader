"""Maps an ITB ``signals.csv`` onto the `backtesting.py <https://kernc.github.io/backtesting.py/>`_
(kernc) library, purely as a secondary reporting/visualization layer — Sharpe ratio, drawdown,
trade list, equity curve plot — on top of ITB's own walk-forward numbers from
``common/backtesting.py``/``scripts/simulate.py``, which remain the source of truth for
correctness (see the project plan's "Backtest engine" decision: ITB's own backtest is
walk-forward-retraining-aware; this is not, it just re-plays the already-computed buy/sell
signal columns).
"""

from __future__ import annotations

import pandas as pd
from backtesting import Backtest, Strategy

from common.io import read_data_file, symbol_data_path


def load_signals_as_ohlc(config: dict) -> pd.DataFrame:
    """Load ``signal_file_name`` and reshape it into the OHLC(V) + signal-column DataFrame
    ``backtesting.py`` expects: capitalized ``Open``/``High``/``Low``/``Close``/``Volume``
    columns and a ``DatetimeIndex``.
    """
    time_column = config["time_column"]
    path = symbol_data_path(config) / config["signal_file_name"]
    df = read_data_file(path, time_column)
    df = df.set_index(time_column)

    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing OHLC column(s) {missing}. signals.csv must include open/high/low/close "
            f"(scripts/signals.py includes them by default)."
        )

    return df.dropna(subset=["Open", "High", "Low", "Close"])


def make_signal_strategy(buy_signal_column: str, sell_signal_column: str) -> type[Strategy]:
    """Build a ``backtesting.py`` Strategy class that just replays precomputed buy/sell boolean
    signal columns — all the actual trade-decision logic already happened in
    ``scripts/signals.py``; this only handles order placement/sizing for the backtest engine.
    """

    class SignalStrategy(Strategy):
        def init(self):
            self.buy_signal = self.I(lambda: self.data.df[buy_signal_column].astype(float), name="buy_signal")
            self.sell_signal = self.I(lambda: self.data.df[sell_signal_column].astype(float), name="sell_signal")

        def next(self):
            if self.buy_signal[-1] and not self.position:
                self.buy()
            elif self.sell_signal[-1] and self.position:
                self.position.close()

    return SignalStrategy


def run_backtest(
    config: dict,
    buy_signal_column: str = "buy_signal_column",
    sell_signal_column: str = "sell_signal_column",
    cash: float = 100_000.0,
    commission: float = 0.0,
) -> tuple[pd.Series, Backtest]:
    """Run the secondary backtest and return (stats, Backtest instance) — the latter so the
    caller can also generate a plot via ``bt.plot(...)``.

    ``commission`` defaults to 0.0 to match v1's fee-less baseline (see
    ``common/backtesting.py``'s docstring); pass a real Binance taker-fee-like value (e.g.
    0.001) once the v2 fee model lands, for an apples-to-apples before/after comparison.

    ``cash`` defaults higher than backtesting.py's own default (10,000) because it sizes
    positions in *whole units* of the traded asset by default: at BTC's price scale, $10,000
    isn't enough headroom for the library's default 99.99%-of-equity sizing to resolve to a
    non-zero position, so orders were silently cancelled for "insufficient margin". $100,000
    comfortably avoids that at typical BTCUSDT prices — override via the CLI ``--cash`` flag if
    your signal/threshold combination still needs more.
    """
    data = load_signals_as_ohlc(config)
    strategy_cls = make_signal_strategy(buy_signal_column, sell_signal_column)
    bt = Backtest(data, strategy_cls, cash=cash, commission=commission, exclusive_orders=True, finalize_trades=True)
    stats = bt.run()
    return stats, bt
