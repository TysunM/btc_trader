"""Example custom feature generator, demonstrating the plugin extension point described in
``common/generators.py``: any function with signature
``fn(df, config, global_config, model_store) -> (df, feature_names)``, referenced from config as
``"generator": "common.my_feature_example:my_feature_example"``, works with zero registration.
Ported from upstream ITB's ``common/my_feature_example.py``.
"""

from __future__ import annotations

import pandas as pd

from common.model_store import ModelStore


def my_feature_example(df: pd.DataFrame, config: dict, global_config: dict, model_store: ModelStore):
    """Add or multiply a column by a constant parameter — config: {"columns": str,
    "function": "add"|"mul", "parameter": number, "names": output column name (optional)}."""
    column_name = config.get("columns")
    if not isinstance(column_name, str) or not column_name:
        raise ValueError(f"'columns' must be a non-empty string, got {column_name!r}")
    if column_name not in df.columns:
        raise ValueError(f"{column_name!r} not found in input data. Existing columns: {df.columns.to_list()}")

    function = config.get("function")
    if function not in ("add", "mul"):
        raise ValueError(f"Unknown function {function!r}. Only 'add' or 'mul' are possible.")

    parameter = config.get("parameter")
    if not isinstance(parameter, (float, int)):
        raise ValueError(f"'parameter' must be a number, got {type(parameter)}")

    names = config.get("names") or f"{column_name}_{function}"

    df[names] = df[column_name] + parameter if function == "add" else df[column_name] * parameter

    return df, [names]
