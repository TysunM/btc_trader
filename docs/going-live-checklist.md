# Going-Live Checklist

This project ships signals-only by default. Nothing here will place a real order without you
deliberately working through every item below — that's by design, not an oversight. Read
`README.md`'s "Scope and safety posture" section first for the mechanics of *why* each gate
exists before using this as a step-by-step list.

**Do not treat a completed checklist as a green light to trade with money you can't afford to
lose.** This confirms the software's safety mechanisms are engaged correctly — it says nothing
about whether the underlying strategy is actually profitable. Re-read the README's "What
'complete' does and doesn't mean" section.

## 1. Validate the strategy itself, not just the plumbing

- [ ] Run `scripts.predict_rolling` + `scripts.simulate` over a much longer/broader date range
      than this project's default configs use (months, not years — genuinely out-of-sample).
- [ ] Re-run with `simulate_model.fee_bps`/`slippage_bps` set to realistic values (see
      `CHANGELOG_V2.md` item 1) — an unprofitable-after-fees strategy is not ready.
- [ ] Confirm the threshold grid was retuned for whatever score distribution your final config
      actually produces (calibration, ensemble, or feature changes all shift this — see
      `CHANGELOG_V2.md`'s "Verification" section for a real example of this mattering).
- [ ] Understand and accept the specific failure modes upstream ITB's own design has: all-in/
      all-out sizing, no max-daily-loss circuit breaker (see README's "Known gaps" section).

## 2. Paper-trade on the live service first

- [ ] Run `service.server` against the config you intend to use, with **no** `trader_binance`
      entry in `output_sets`, for at least several days/weeks.
- [ ] Confirm the paper simulator's (`outputs/notifier_trades.py`) logged transactions and the
      offline backtest's numbers are directionionally consistent — a live/offline mismatch here
      means something (data latency, a subtle feature-computation difference) is wrong, and you
      should not proceed until you understand why.
- [ ] If using stop-loss/take-profit (`trade_model.stop_loss_pct`/`take_profit_pct`), confirm in
      the logs that they actually fire when expected during the paper period.

## 3. Prepare credentials yourself — the assistant that built this never touches this step

- [ ] Create your own Binance API key. Do **not** grant withdrawal permission.
- [ ] Populate `.env` (never commit it) with `BINANCE_API_KEY`/`BINANCE_API_SECRET`.
- [ ] Create your own Telegram bot via @BotFather; populate `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`.
- [ ] Start with a Binance sub-account or a small dedicated balance, not your main account.

## 4. Review the highest-risk file yourself

- [ ] Read `outputs/trader_binance.py` in full, not just this checklist. Confirm you understand
      and accept: all-in/all-out position sizing, unconditional in-flight-order cancellation
      after one scheduler interval, no min-notional/lot-size pre-validation beyond basic
      rounding, no idempotency guard against double submission.
- [ ] Confirm `trade_model.no_trades_only_data_processing` and `trade_model.
      test_order_before_submit` are set the way you intend for your first real run.

## 5. Flip the three independent gates, deliberately, in order

- [ ] Add a `trader_binance` entry to your config's `output_sets`.
- [ ] Set `trade_model.simulate_order_execution: false` explicitly in that config (the project
      default is `true` — simulated — on purpose).
- [ ] Set the `ITB_ALLOW_LIVE_TRADING=1` environment variable (not in the config file).
- [ ] Start the server with `--i-understand-live-trading-risk`.
- [ ] Watch `server.log` and `orders_dry_run.log` closely for at least the first several ticks.

## 6. Ongoing

- [ ] Monitor `orders_dry_run.log` and `server.log` regularly, not just at startup.
- [ ] Have a plan for manually flattening your position and stopping the server — don't assume
      the automated safety mechanisms (circuit breaker, SL/TP) cover every scenario.
- [ ] Revisit item 1 periodically — markets change regime; a backtest from months ago is not a
      standing guarantee.
