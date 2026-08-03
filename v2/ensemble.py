"""Cross-algorithm ensemble signal (v2 quant pass, item 8 — see CHANGELOG_V2.md, lowest priority
of the eight items: a refinement, not a gap-fix).

Averaging/stacking predictions across algorithm families (e.g. gb+lc+svc instead of a single
model per label) reduces variance from any one model's idiosyncrasies. This is a ``signal_sets``
generator using the existing zero-registration plugin mechanism — it should run *before* the
``combine`` entry in ``signal_sets``, blending each label's multiple per-algorithm score columns
(e.g. ``high_30_lc``, ``high_30_gb``, ``high_30_svc``) into one column that ``combine`` then
differences against the blended low-label score, exactly as it already does for a single
algorithm's two columns.
"""

from __future__ import annotations

import pandas as pd

from common.model_store import ModelStore


def generate_ensemble_score(df: pd.DataFrame, config: dict, global_config: dict, model_store: ModelStore):
    """config: {"columns": [<per-algorithm score columns for one label>], "names": <output
    column name>, "method": "mean" (default) | "median"}."""
    columns = config.get("columns")
    if not columns or not isinstance(columns, list) or len(columns) < 2:
        raise ValueError(f"'columns' must be a list of 2+ score columns to ensemble, got {columns!r}")

    method = config.get("method", "mean")
    name = config.get("names")
    if not name:
        raise ValueError("'names' (output column name) is required.")

    if method == "mean":
        df[name] = df[columns].mean(axis=1, skipna=True)
    elif method == "median":
        df[name] = df[columns].median(axis=1, skipna=True)
    else:
        raise ValueError(f"Unknown ensemble method {method!r}. Available: 'mean', 'median'.")

    return df, [name]
