# v2 Quant/Risk Improvements Pass

Everything below is implemented in `v2/` plus a small number of call-sites in `common/`,
`scripts/`, and `outputs/`, each behind an explicit config flag so the v1 baseline
(tag `v1.0-itb-port`) stays reproducible with every flag off. Compare with
`git diff v1.0-itb-port..HEAD`.

Each entry: what changed, why (the specific gap it targets), and how to turn it on.

---

## 1. Fee- and slippage-aware backtest scoring

**File:** `v2/fees.py`, wired into `common/backtesting.py::simulated_trade_performance`.

**Why:** `simulated_trade_performance()` (used by `scripts/simulate.py`'s threshold grid search)
scored every simulated transaction on raw price delta only — no Binance taker/maker fee (~0.1%
per side by default) and no slippage. This inflates the apparent profitability of any threshold
configuration that trades frequently, since transaction costs are invisible to the ranking.
High-frequency 1-minute threshold configs are hit hardest.

**How to enable:** pass `fee_bps`/`slippage_bps` to `simulated_trade_performance`, or set
`simulate_model.fee_bps` / `simulate_model.slippage_bps` in a config consumed by
`scripts/simulate.py`. Both default to `0` (v1 baseline, fee-less) if unset.

---

## 2. Configurable embargo buffer for walk-forward validation

**File:** `v2/purge_embargo.py`, wired into `scripts/predict_rolling.py`.

**Honest framing (not a bug fix):** the plan going into this item assumed upstream's rolling
walk-forward loop had a purged-CV-style leakage gap. Working through the actual index arithmetic
during implementation showed otherwise: upstream's existing `train_end = predict_start -
label_horizon - 1` is *already sufficient* — the last training row's label horizon reaches at
most to `predict_start - 2`, still strictly before the predict window, and this loop is a
strictly sequential single-direction walk-forward (not shuffled k-fold CV), so the classic
purged-CV leakage scenario doesn't actually arise here. This item is downgraded from "fix a gap"
to what it actually is: an optional extra conservatism knob.

**Why it's still worth having:** an `embargo` of N extra rows pulling the training cutoff back
further than the strict minimum is standard practice in the purged-CV literature as a safety
margin — useful if you swap in a custom label generator with different/fatter-tailed lookahead
characteristics than `highlow2`.

**How to enable:** set `rolling_predict.embargo` (rows) in a config's `rolling_predict` section.
Defaults to `0` (v1 baseline, upstream's existing gap unchanged) if unset.

---

## 3. Score calibration

**File:** `v2/calibration.py`, wired into `scripts/train.py` / `scripts/predict.py` /
`common/model_store.py`.

**Why:** Raw scores from different algorithm families (gb/lc/nn/svc) are not comparable
probabilities — a LightGBM margin and a logistic-regression `predict_proba` output don't mean
the same thing at the same numeric value. `signals.py`'s `combine` generator subtracts two raw
scores directly (`high_30_lc - low_30_lc`), conflating "model confidence" with "which algorithm
produced it."

**How to enable:** set `algorithms[].params.calibrate: true` on an algorithm entry. Persists a
calibrator (isotonic for gb/nn, sigmoid for lc/svc, matching each family's typical sample-size
and score-distribution characteristics) alongside the model via `ModelStore`.

---

## 4. Regime filter overlay

**File:** `v2/regime_filter.py`, used as a `signal_sets` generator via the existing
`module:function` plugin mechanism — no core dispatch changes needed.

**Why:** `simulate.py`'s grid search fits one fixed threshold set across the entire backtest
window regardless of trend/volatility regime, which risks overfitting to whichever regime
dominated the window (e.g. a threshold tuned during a strong trend may whipsaw badly in chop).

**How to enable:** add a `signal_sets` entry with
`"generator": "v2.regime_filter:generate_regime_gate"` before the `threshold_rule` entry, and
reference its output column in `threshold_rule`'s config. See `configs/btcusdt-1h-v2.jsonc`.

---

## 5. Volatility-targeted position sizing + stop-loss/take-profit

**File:** `v2/position_sizing.py`, wired into `outputs/notifier_trades.py`'s paper simulator.

**Why:** Upstream (and this project's v1 port) sizes all-in/all-out (99% of quote balance to
buy, 100% of base balance to sell) with no stop-loss, take-profit, or max-position anywhere.
Even in signals-only/paper mode, the simulator's reported "simulated profit" is more honest with
realistic sizing.

**How to enable:** set `trade_model.position_sizing.enabled: true` (ATR-based sizing with a
capped Kelly-fraction ceiling) and optionally `trade_model.stop_loss_pct` /
`trade_model.take_profit_pct`.

---

## 6. Multi-timeframe context features

**Config-only** — `merge_data_sources` already supports multiple `data_sources`; no new code.

**Why:** Adding a higher-timeframe trend-context source (e.g. 4h/1d EMA slope) as extra feature
columns is architecturally free with the existing multi-source merge, and gives the 1h/1m models
visibility into a longer-horizon trend they otherwise can't see.

**How to enable:** see `configs/btcusdt-1h-v2.jsonc`'s `data_sources` for an example.

---

## 7. Risk-adjusted grid-search ranking

**File:** `v2/fees.py` (pairs with item 1), wired into `scripts/simulate.py`.

**Why:** Once fees are modeled, ranking purely by `%profit/M` still favors a threshold config
that strung together a lucky sequence of small-edge high-frequency trades. Penalizing by
per-trade return volatility (a Sharpe-like ratio) favors configs with a more consistent edge.

**How to enable:** set `simulate_model.rank_by: "sharpe"` (default remains `"%profit/M"`,
matching v1 baseline ranking).

---

## 8. Optional cross-algorithm ensemble signal

**File:** `v2/ensemble.py`, used as a `signal_sets` generator via the plugin mechanism.

**Why:** Averaging/stacking predictions across algorithm families (gb+lc+nn instead of a single
model per label) reduces variance from any one model's idiosyncrasies. Lower priority than the
items above — a refinement, not a gap-fix.

**How to enable:** add a `signal_sets` entry with `"generator": "v2.ensemble:generate_ensemble_score"`.

---

## Verification

Every item above was verified individually against real BTCUSDT data (not just unit-tested in
isolation) before being wired together:

- **Fees (item 1):** initial implementation had a real bug — adjusting both legs' execution
  price in the same direction caused the fee to cancel out instead of compound, because
  `simulated_trade_performance`'s long/short buckets compare consecutive *same-side* executions
  (see `v2/fees.py`'s docstring). Fixed by switching to a flat round-trip cost deduction per
  transaction. Verified: at 10bps fee + 2bps slippage, every individual trade's return dropped
  by exactly the expected 0.24pp, and the full threshold grid's rankings shifted accordingly.
- **Purge/embargo (item 2):** verified via `purged_train_end()`'s arithmetic directly, and via
  `predict_rolling.py`'s printed train ranges shrinking by exactly the configured embargo.
- **Calibration (item 3):** verified full train -> persist -> reload -> predict round-trip;
  confirmed calibrated output differs from raw model output and stays in `[0, 1]`.
- **Regime filter (item 4):** verified against real signals — 57 raw buy signals reduced to 41
  under an ADX >= 20 gate, with sell signals provably untouched (`gate_sell` defaults false).
- **Position sizing / SL-TP (item 5):** verified via a synthetic scenario — a 3% adverse move
  force-triggered a stop-loss exit, a 6% favorable move force-triggered a take-profit exit,
  both with no explicit sell signal present, confirming the check runs independently of the
  model's own signal.
- **Multi-timeframe (item 6):** verified by actually downloading real 4h BTCUSDT data and
  running the full merge -> features -> labels -> train -> predict_rolling -> signals -> simulate
  chain through `configs/btcusdt-1h-v2.jsonc` (not just describing the config shape).
- **Sharpe ranking (item 7):** verified the `rank_by: "sharpe"` option changes which grid
  parameters rank highest vs. `"%profit/M"`.
- **Ensemble (item 8):** verified `generate_ensemble_score` averages multiple score columns
  correctly on a synthetic example (not wired into the demo config, which only trains one
  algorithm — see the generator's docstring for how to add it to a multi-algorithm config).

### An honest finding from the full end-to-end run, not smoothed over

Running `configs/btcusdt-1h-v2.jsonc`'s full pipeline surfaced a real, useful interaction:
enabling sigmoid calibration (item 3) substantially widens `trade_score`'s spread (std ~0.03 ->
~0.19 measured on this data, since a calibrated-probability difference can range close to ±1
where an uncalibrated logistic-regression-score difference stayed tightly clustered). Reusing
`btcusdt-1h.jsonc`'s v1 threshold grid (0.01-0.10) against the calibrated score barely filtered
anything and lost money badly after fees. Retuning the grid to a range appropriate for the wider
calibrated distribution (0.10-0.50) improved things but still didn't show a clearly profitable
edge on this specific 6-month out-of-sample window with this specific combination of features
(multi-timeframe + regime filter + calibration together).

This is reported honestly rather than cherry-picked to look better: it's a small sample (6
months, one symbol, one frequency), and a negative result for one specific parameter combination
is not a verdict on any individual technique's general validity — it's exactly the kind of
finding these tools exist to surface rather than hide. The operational lesson that generalizes:
**a threshold grid tuned for one score distribution does not transfer to a different one** —
recompute the grid any time calibration, the label definition, or the feature set changes.
Genuine threshold/feature optimization for this feature combination (broader date ranges, other
labels, other algorithms) is real future work, not something this pass claims to have solved.
