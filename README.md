# ITB-BTCUSDT

A port of the open-source [Intelligent Trading Bot](https://github.com/asavinov/intelligent-trading-bot)
(ITB) architecture, scoped to **BTCUSDT spot on Binance**, followed by a separate, clearly-tagged
pass of quant/risk improvements. See `CHANGELOG_V2.md` for the itemized diff from stock ITB
behavior — every v2 change is opt-in via config, so `git diff v1.0-itb-port..v2.0-quant-pass`
is the literal, reviewable delta.

**Status: all 6 planned phases complete.** Offline pipeline, full ML/generator fidelity, live
signals-only service, secondary backtest reporting, the v2 quant pass, and test hardening are
all built and verified against real Binance market data. See "What 'complete' does and doesn't
mean" below before treating this as a finished trading product.

## The core invariant

> The same (derived) features must be computed identically whether running in offline (batch,
> training) mode or online (live, streaming) mode.

Every script (`scripts/*.py`) and the live service (`service/server.py`) route through the same
generator-dispatch functions in `common/generators.py`. This is not a convenience — it's what
makes the backtest numbers trustworthy: if online and offline computed features differently, a
backtest could look good on data the live model never actually sees the same way.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). Python 3.12 is pinned and managed by `uv` itself —
you don't need a system Python 3.12 install.

```
uv sync --extra dev
cp .env.example .env   # fill in only what you need — everything is optional for signals-only use
```

Optional extras:
- `--extra nn` — TensorFlow/Keras, only needed once you configure an `nn` algorithm entry.
- `--extra reporting` — the `backtesting.py` secondary reporting layer.
- `--extra diagram` — matplotlib/seaborn, only needed for the diagram Telegram notifier.

See `docs/ta-lib-windows.md` if you specifically want the real `talib` backend instead of the
default `pandas_ta`.

## Quick start

```
# BTCUSDT 1h, logistic regression -- the fastest path to a real backtest
uv run python -m scripts.download -c configs/btcusdt-1h.jsonc
uv run python -m scripts.merge    -c configs/btcusdt-1h.jsonc
uv run python -m scripts.features -c configs/btcusdt-1h.jsonc
uv run python -m scripts.labels   -c configs/btcusdt-1h.jsonc
uv run python -m scripts.train    -c configs/btcusdt-1h.jsonc
uv run python -m scripts.predict_rolling -c configs/btcusdt-1h.jsonc   # walk-forward backtest
uv run python -m scripts.signals  -c configs/btcusdt-1h.jsonc
uv run python -m scripts.simulate -c configs/btcusdt-1h.jsonc          # ranked threshold grid

# Or all at once, plus sanity checks on every output file:
uv run python -m scripts.smoke_test -c configs/btcusdt-1h.jsonc

# A nicer report (Sharpe, drawdown, equity curve) on top of the signals.csv above:
uv run python -m reporting.run_report -c configs/btcusdt-1h.jsonc

# Live, signals-only (Telegram + logs, no order execution):
uv run python -m service.server -c configs/btcusdt-1h.jsonc
```

`binance.com` blocks requests from some regions; `configs/*.jsonc` default to `binance.us` via
`client_args: {"tld": "us"}` — remove that line if you're somewhere `binance.com` serves directly.

Other ready-to-run configs: `configs/btcusdt-1m.jsonc` (1-minute, matches upstream ITB's own
default frequency), `configs/btcusdt-1h-nn.jsonc` (Keras MLP instead of logistic regression),
`configs/btcusdt-1h-v2.jsonc` (every v2 quant improvement turned on — read its header comments
and `CHANGELOG_V2.md` before running; it needs its own `features`/`labels` regeneration and a
retuned threshold grid, both explained there).

## Testing

```
uv run pytest tests/ -v              # ~150 unit/integration tests, no network, seconds to run
uv run python -m scripts.smoke_test -c configs/btcusdt-1h.jsonc   # real network, real pipeline
```

Run the smoke test after any change that touches the pipeline mechanics (feature/label/signal
generators, the classifier dispatch, the walk-forward loop) — the unit tests check individual
pieces in isolation; the smoke test is the only thing that proves they still fit together
correctly against real market data.

## Scope and safety posture (read this before touching `trade_model` or `output_sets`)

This project is **signals-only** by design: it computes buy/sell scores and sends Telegram
notifications / logs, and does not place real orders. The order-execution code path
(`outputs/trader_binance.py`) is ported for architectural completeness but is reachable only if
**all** of the following are true:

1. A config's `output_sets` explicitly includes a `trader_binance` entry (none of this repo's
   shipped configs do).
2. The `ITB_ALLOW_LIVE_TRADING=1` environment variable is set — checked in
   `common/generators.py::output_feature_set`, deliberately outside the JSONC config file so an
   accidental config edit alone can never enable it. Only the exact string `"1"` opens the gate.
3. `service/server.py` was started with `--i-understand-live-trading-risk`.

Even then, per-order config flags still apply: `trade_model.simulate_order_execution` defaults to
`true` in this project (a deliberate flip from upstream ITB's risky-by-default `false` — see
`common/config.py`), so a real order still won't be submitted unless that's explicitly turned off
too. Every order that would be submitted is also written to a local, gitignored
`orders_dry_run.log` audit trail regardless of these gates. A tick-failure circuit breaker
(`max_consecutive_tick_failures`, default 5) halts the scheduler rather than retrying forever
against a broken connection — see `service/server.py`'s module docstring for this and the
health-check-wiring fix vs. upstream.

All of this is exercised by `tests/test_trader_binance.py`, including a test that walks every
wrong value of `ITB_ALLOW_LIVE_TRADING` (`"true"`, `"yes"`, `"0"`, `""`) and confirms none of
them open the gate.

Going live is always a deliberate, multi-step, human action — nothing in this repo will ever
place a real order using defaults alone. See `docs/going-live-checklist.md` for the full
sequence if you ever actually intend to.

**Never commit `.env`.** API keys and Telegram tokens are only ever read from your local `.env`
or OS environment — never hardcoded, never written to a config file as a literal value. I (the
assistant that built this) never handled, entered, or stored real credentials at any point.

## What "complete" does and doesn't mean

All 6 planned phases are built, wired together, and verified — imports cleanly, runs end to end
against real Binance market data, ~150 passing tests, every safety gate exercised. That is a
genuinely different, higher bar than "the code exists."

What it does **not** mean: that this strategy is profitable, or that a passing backtest predicts
future performance. Concretely:

- The default configs' backtests run on a small sample (months, one symbol, one exchange). See
  `CHANGELOG_V2.md`'s "Verification" section for an example of a specific parameter combination
  that looked fine mechanically but lost money once fees were modeled — that's not a bug, that's
  what an honest backtest is supposed to be able to show you.
- No amount of historical backtesting guarantees future results. Markets change regime;
  overfitting to a backtest window is a real risk even with the walk-forward/purge/fee tooling
  this project includes specifically to reduce (not eliminate) that risk.
- Threshold/feature/algorithm selection here is illustrative, not a tuned, validated trading
  strategy. Real strategy development would need much broader date ranges, out-of-sample
  validation across different market regimes, and — before any real money — a paper-trading
  period on the live service (still signals-only in this repo) to confirm the offline backtest
  numbers actually reproduce online.
- I am not a licensed financial advisor and nothing here is investment advice.

## Phases

| Phase | Scope |
|---|---|
| 0 | Repo scaffolding, config loader, TA backend adapter |
| 1 | MVP offline pipeline: download → merge → features → labels → train → predict → signals → simulate, BTCUSDT 1h, one algorithm |
| 2 | Full offline fidelity: all feature/label generators, all four ML classifiers, both 1m/1h configs |
| 3 | Online service, signals-only (Telegram notifications, no order execution) |
| 4 | Secondary backtest reporting layer (`backtesting.py`) |
| 5 | v2 quant/risk improvements pass (fees/slippage, calibration, regime filter, position sizing, multi-timeframe, embargo, Sharpe ranking, ensemble) — see `CHANGELOG_V2.md` |
| 6 | Test hardening (~150 tests), docs, guardrail audit |

## Layout

```
common/     shared pipeline code: config, generators dispatch, feature/label/signal generators,
            ML classifiers, model persistence, the online Analyzer state object
inputs/     Binance data collector
outputs/    Telegram notifiers, paper-trading simulator, (gated-off) real trader
scripts/    one CLI entrypoint per offline pipeline stage, plus smoke_test.py
service/    the live scheduler/service (app_state.py, server.py)
reporting/  secondary backtesting.py-based report generator
v2/         quant/risk improvements, isolated from the v1 port (see CHANGELOG_V2.md)
configs/    per-symbol/frequency JSONC config files
tests/      pytest suite (no network) + the smoke-test harness (real network, in scripts/)
```

## Known gaps / deliberately out of scope

- **Yahoo and MT5 venues**: not ported (this project targets Binance spot only). MT5 in
  particular was surveyed during research and found to have a live ordering bug upstream — not
  relevant here since it was never ported.
- **`tsfresh`/`depth` feature generators**: not ported — optional upstream dependency needing an
  older Python (tsfresh) or order-book depth data this project doesn't collect (depth).
  `itbstats` covers similar statistical-feature ground without the tsfresh dependency.
- **Min-notional/lot-size/precision validation, idempotency guards against double order
  submission, max-daily-loss circuit breaker**: upstream gaps in `trader_binance.py`, ported
  faithfully (not fixed) since that file is gated off by design — see the safety section above.
  Would need addressing before ever considering live execution.
- **Position sizing is advisory, not enforced**: the v2 ATR-based sizing (`v2/position_sizing.py`)
  computes and logs a recommendation but doesn't change the paper simulator's all-in/all-out
  behavior — see `CHANGELOG_V2.md` item 5 for why that's a larger rewrite than this pass scoped in.
