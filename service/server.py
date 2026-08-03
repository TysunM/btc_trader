"""The live, signals-only service. ``python -m service.server -c configs/btcusdt-1h.jsonc``.

Every tick: (1) fetch new klines and append them to the :class:`~common.analyzer.Analyzer`'s
rolling window, (2) recompute features/predictions/signals for the dirty tail via
``Analyzer.analyze()`` (offloaded to a thread pool since it's CPU-bound sync code), (3) dispatch
each configured ``output_sets`` entry (Telegram notifications, the paper-trading simulator, and
— only if every gate in the README's "Safety" section is open — real order submission).

Deviations from upstream ITB's ``service/server.py`` (both discussed in the project plan):

1. **Health-check wiring fix.** Upstream calls ``await health_check_fn()`` when a problem is
   already flagged, but discards its return value — so ``server_status`` never actually gets
   updated from a fresh check; ``data_provider_problems_exist()`` just keeps seeing whatever was
   set at the *previous* failure. This port assigns the result back to ``state.server_status``.
2. **Circuit breaker.** Upstream logs-and-continues forever on any tick failure, with no
   backoff or halt. This port tracks consecutive tick failures and halts the scheduler after
   ``config['max_consecutive_tick_failures']`` (default 5), requiring a manual restart rather
   than retrying against a possibly-broken connection indefinitely.

Also uses the instance-based ``AppState`` (``service/app_state.py``) instead of upstream's
class-level global ``App``, threading ``state`` explicitly through every function instead of
relying on ``from service.App import *``.
"""

from __future__ import annotations

import asyncio
import logging
import os

import click
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from common.config import load_config, require_fields
from common.generators import output_feature_set
from common.model_store import ModelStore
from common.analyzer import Analyzer
from common.types import Venue
from common.utils import freq_to_cron_trigger, now_timestamp, pandas_get_interval
from inputs import get_collector_functions
from outputs.notifier_trades import load_last_transaction
from service.app_state import AppState, get_state, set_state

log = logging.getLogger("server")


async def main_collector_task(state: AppState) -> int:
    """Fetch new data and append it to the analyzer. Returns 0 on success, 1 on any failure."""
    venue = Venue(state.config.get("venue", "binance"))
    fetch_klines_fn, health_check_fn = get_collector_functions(venue)

    freq = state.config["freq"]
    start_ts, end_ts = pandas_get_interval(freq)
    now_ts = now_timestamp()
    log.info(f"===> Start collector task. Timestamp {now_ts}. Interval [{start_ts},{end_ts}].")

    if state.data_provider_problems_exist():
        # Health-check wiring fix (see module docstring): capture and actually use the result.
        state.server_status = await health_check_fn()
        if state.data_provider_problems_exist():
            log.error("Data provider server has problems. Skipping this tick.")
            return 1

    last_kline_dt = state.analyzer.get_last_kline_dt()
    dfs = await fetch_klines_fn(state.config, last_kline_dt)
    if dfs is None:
        log.error("Problem getting data from the server. Will try next tick.")
        state.server_status = 1
        return 1
    state.server_status = 0

    try:
        state.analyzer.append_data(dfs)
    except Exception as e:
        log.error(f"Error appending data to analyzer: {e}")
        return 1

    log.info("<=== End collector task.")
    return 0


async def main_task(state: AppState) -> None:
    """Executed once per scheduled interval: collect -> analyze -> dispatch outputs."""
    try:
        res = await main_collector_task(state)
    except Exception as e:
        log.error(f"Error in main_collector_task: {e}")
        res = 1
    if res:
        _record_tick_outcome(state, ok=False)
        return

    try:
        await state.loop.run_in_executor(None, state.analyzer.analyze)
    except Exception as e:
        log.error(f"Error in analyzer.analyze(): {e}")
        _record_tick_outcome(state, ok=False)
        return

    for output_set in state.config.get("output_sets", []):
        try:
            await output_feature_set(state.analyzer.df, output_set, state.config, state.model_store)
        except Exception as e:
            log.error(f"Error in output generator {output_set.get('generator')!r}: {e}")
            _record_tick_outcome(state, ok=False)
            return

    _record_tick_outcome(state, ok=True)


def _record_tick_outcome(state: AppState, ok: bool) -> None:
    """Circuit breaker (see module docstring): halt the scheduler after too many consecutive
    tick failures instead of retrying forever."""
    if ok:
        state.consecutive_failures = 0
        return

    state.consecutive_failures += 1
    max_failures = state.config.get("max_consecutive_tick_failures", 5)
    if state.consecutive_failures >= max_failures:
        log.error(
            f"{state.consecutive_failures} consecutive tick failures -- halting the scheduler. "
            f"Investigate and restart manually; this will not retry further on its own."
        )
        if state.sched and state.sched.running:
            state.sched.shutdown(wait=False)
        if state.loop and state.loop.is_running():
            state.loop.stop()


@click.command()
@click.option("--config_file", "-c", type=click.Path(exists=True), required=True, help="Path to a config .jsonc file")
@click.option(
    "--i-understand-live-trading-risk",
    "confirmed_live_trading",
    is_flag=True,
    default=False,
    help=(
        "Required, in addition to output_sets containing a 'trader_binance' entry AND the "
        "ITB_ALLOW_LIVE_TRADING=1 environment variable, before the server will place real "
        "orders. See the README 'Safety / Guardrails' section."
    ),
)
def start_server(config_file: str, confirmed_live_trading: bool) -> None:
    logging.basicConfig(
        filename="server.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    config = load_config(config_file)
    require_fields(config, ["symbol", "freq", "venue", "data_folder", "train_features", "labels"])
    config["train"] = False  # the server only predicts, it never trains

    #
    # Live-trading gate: three independent checks must all pass (see this module's docstring
    # and outputs/trader_binance.py's docstring for the full rationale).
    #
    wants_live_trading = any(o.get("generator") == "trader_binance" for o in config.get("output_sets", []))
    if wants_live_trading:
        if os.environ.get("ITB_ALLOW_LIVE_TRADING") != "1":
            log.error(
                "output_sets includes 'trader_binance' but ITB_ALLOW_LIVE_TRADING=1 is not set "
                "in the environment. Refusing to start. See the README 'Safety' section."
            )
            return
        if not confirmed_live_trading:
            log.error(
                "output_sets includes 'trader_binance' and ITB_ALLOW_LIVE_TRADING=1 is set, but "
                "--i-understand-live-trading-risk was not passed. Refusing to start."
            )
            return
        log.warning(
            "LIVE TRADING ENABLED. trader_binance may submit real orders, subject to its own "
            "config-level test_order_before_submit/simulate_order_execution/"
            "no_trades_only_data_processing gates."
        )

    try:
        venue = Venue(config.get("venue", "binance"))
    except ValueError:
        log.error(f"Unsupported venue {config.get('venue')!r}. Supported: {[v.value for v in Venue]}")
        return

    log.info(f"Initializing server. Venue: {venue.value}. Symbol: {config['symbol']}. Freq: {config['freq']}")

    if venue == Venue.BINANCE:
        client_params = {}
        if config.get("append_overlap_records"):
            client_params["append_overlap_records"] = config["append_overlap_records"]
        client_args = dict(api_key=config.get("api_key"), api_secret=config.get("api_secret"))
        client_args.update(config.get("client_args", {}))
        from inputs.collector_binance import init_client

        init_client(client_params, client_args)

    model_store = ModelStore(config)
    model_store.load_models()
    analyzer = Analyzer(config, model_store)

    state = AppState(config)
    state.model_store = model_store
    state.analyzer = analyzer
    state.transaction = load_last_transaction(config)
    set_state(state)

    state.loop = asyncio.new_event_loop()

    #
    # Cold start: two collector passes (the first, potentially slow due to a large initial
    # history fetch, may miss klines that closed while it was running -- the second catches up),
    # then one full analyze() over the whole history.
    #
    try:
        state.loop.run_until_complete(main_collector_task(state))
        state.loop.run_until_complete(main_collector_task(state))
        state.analyzer.analyze()
    except Exception as e:
        log.error(f"Problem during initial data collection: {e}")

    if state.data_provider_problems_exist():
        log.error("Problems during initial data collection. Not starting the scheduler.")
        return

    log.info("Finished initial data collection (cold start).")

    if wants_live_trading:
        from outputs import get_trader_functions

        trader_funcs = get_trader_functions(venue)
        try:
            state.loop.run_until_complete(trader_funcs["update_trade_status"](config))
        except Exception as e:
            log.error(f"Problem syncing trade status: {e}")
        if state.problems_exist():
            log.error("Problems during trade status sync. Not starting the scheduler.")
            return
        log.info(
            f"Trade status synced. Balance: base={state.account_info.base_quantity} "
            f"quote={state.account_info.quote_quantity}"
        )

    state.sched = AsyncIOScheduler(event_loop=state.loop)
    trigger = freq_to_cron_trigger(config["freq"])
    state.sched.add_job(main_task, trigger=trigger, id="main_task", args=[state])
    state.sched.start()

    log.info("Scheduler started.")

    try:
        state.loop.run_forever()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt.")
    finally:
        log.info("Shutting down...")
        if state.sched and state.sched.running:
            state.sched.shutdown()
        if state.loop.is_running():
            state.loop.stop()
        state.loop.close()
        if venue == Venue.BINANCE:
            from inputs.collector_binance import close_client

            close_client()
        log.info("Connection closed.")


if __name__ == "__main__":
    start_server()
