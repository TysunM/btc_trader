# TA-Lib on Windows

The real `talib` package wraps a native C library that is notoriously painful to build on
Windows (no official wheels for recent CPython versions on PyPI as of writing). This project
defaults to `pandas_ta` (pure Python, installs via plain `pip`/`uv`, no compiler needed) via the
`ITB_TA_BACKEND` env var, so you never have to deal with this unless you specifically want it.

## Do you need real talib at all?

No. `ITB_TA_BACKEND=pandas_ta` (the default) is sufficient for every phase of this project.
Every feature generator goes through `common/ta_adapter.py`, which only implements the small set
of indicators ITB's own sample configs actually use, natively in pandas/numpy — no third-party
indicator library dependency for the primitives that matter for parity with upstream ITB.

## If you want real talib anyway

The most reliable path on Windows is `conda-forge`, which ships precompiled binaries:

```
conda install -c conda-forge ta-lib
```

Then set `ITB_TA_BACKEND=talib` in `.env`. Note: exact-value parity between `pandas_ta`/native
and real `talib` output is not guaranteed bar-for-bar (different smoothing conventions on the
first `window` warm-up rows in some indicators) — this only matters if you're trying to
reproduce upstream ITB's numbers exactly, not for this project's own correctness.
