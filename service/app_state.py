"""Instance-based replacement for upstream ITB's ``service/App.py`` class-level global state.

Deviation from upstream: ITB's ``App`` is a class with class-level (not instance) attributes —
the project plan flagged this as (a) not thread-safe, (b) unable to run more than one
symbol/config per process, and (c) awkward to test (shared mutable state with no clean reset
between test cases, since it lives on the class itself rather than an object). This module
instead defines :class:`AppState` as a real class constructed explicitly once, in
``service/server.py:start_server()``, with :func:`get_state`/:func:`set_state` as the single,
explicit access point other modules (notifiers, the trader) use to reach it — rather than
``from service.App import *`` implicitly pulling in shared mutable class attributes.

This remains a process-wide singleton, not a fully composable multi-instance design — but
upstream's own docs state only one symbol is supported per running server instance anyway
(``docs/server.md``), so running two configs concurrently still means two OS processes, exactly
as documented upstream. The improvement here is explicit construction and typing, not
multi-tenancy within a single process.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from common.analyzer import Analyzer
from common.model_store import ModelStore
from common.types import AccountBalances


class AppState:
    def __init__(self, config: dict):
        self.config = config

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.sched: Optional[AsyncIOScheduler] = None
        self.analyzer: Optional[Analyzer] = None
        self.model_store: Optional[ModelStore] = None

        # Health flags: 0 = OK, non-zero = a problem exists. Checked before each tick's collector
        # task and gate output dispatch (see server.py's health-check wiring fix).
        self.error_status = 0
        self.server_status = 0
        self.account_status = 0
        self.trade_state_status = 0

        # Trading state machine -- only meaningful once trader_binance is actually enabled (see
        # the README's "Safety / Guardrails" section; off by every default in this project).
        self.status: Optional[str] = None  # "SOLD" | "BOUGHT" | "BUYING" | "SELLING"
        self.order: Optional[dict] = None
        self.order_time: Optional[int] = None
        self.transaction: Optional[dict] = None
        self.account_info = AccountBalances()

        # Circuit breaker (deviation from upstream, which logs-and-continues forever on any tick
        # failure): consecutive tick failures, reset to 0 on any tick that completes end to end.
        self.consecutive_failures = 0

    def data_provider_problems_exist(self) -> bool:
        return self.error_status != 0 or self.server_status != 0

    def problems_exist(self) -> bool:
        return (
            self.error_status != 0
            or self.server_status != 0
            or self.account_status != 0
            or self.trade_state_status != 0
        )


_current_state: Optional[AppState] = None


def get_state() -> AppState:
    if _current_state is None:
        raise RuntimeError(
            "AppState has not been initialized. set_state() must be called once, normally by "
            "service/server.py:start_server(), before any code that calls get_state()."
        )
    return _current_state


def set_state(state: AppState) -> None:
    global _current_state
    _current_state = state
