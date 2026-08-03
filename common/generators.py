"""The plugin dispatch mechanism: every ``feature_sets``/``label_sets``/``signal_sets``/
``output_sets``/``train_feature_sets`` config entry names a *generator*, resolved here to
either a built-in function or — via ``common.utils.resolve_generator_name`` — an arbitrary
``module.path:function_name`` custom function with signature
``fn(df, config, global_config, model_store) -> (df, feature_names)``.

This is the single invariant that makes ITB's offline/online parity guarantee hold: both the
batch scripts (``scripts/*.py``) and the live service (``service/server.py``) call
:func:`generate_feature_set`/:func:`predict_feature_set`/:func:`output_feature_set` — never
duplicate feature-computation logic in two places.

Ported from upstream ``common/generators.py``. Some ``elif`` branches raise
``NotImplementedError`` for generators not yet ported (tracked per-phase in the project plan)
rather than being silently omitted, so the roadmap for extending this dispatcher is visible
in the code itself.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pandas as pd

from common.gen_features import generate_features_itblib, generate_features_itbstats, generate_features_talib
from common.gen_labels_highlow import generate_labels_highlow, generate_labels_highlow2
from common.gen_labels_topbot import generate_labels_topbot, generate_labels_topbot2
from common.gen_signals import (
    generate_combine_scores,
    generate_smoothen_scores,
    generate_threshold_rule,
    generate_threshold_rule2,
)
from common.model_store import ModelStore
from common.utils import find_algorithm_by_name, resolve_generator_name

log = logging.getLogger("generators")

_NOT_YET_PORTED = {
    "depth": "not planned (order-book depth data, out of scope for spot BTCUSDT)",
    "tsfresh": "not planned (optional upstream dependency, needs an older Python; itbstats covers the same statistical-feature ground without it)",
}


def generate_feature_set(
    df: pd.DataFrame, fs: dict, config: dict, model_store: ModelStore, last_rows: int = 0
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the generator named in ``fs['generator']`` to (a column-prefix-selected slice of) df."""
    cp = fs.get("column_prefix")
    if cp:
        prefix = cp + "_"
        f_cols = [c for c in df.columns if c.startswith(prefix)]
        f_df = df[f_cols].rename(columns=lambda x: x[len(prefix):] if x.startswith(prefix) else x)
    else:
        f_df = df[df.columns.to_list()]

    generator = fs.get("generator")
    gen_config = fs.get("config", {})

    if generator in _NOT_YET_PORTED:
        raise NotImplementedError(f"Generator {generator!r} is not ported yet ({_NOT_YET_PORTED[generator]}).")

    if generator == "talib":
        features = generate_features_talib(f_df, gen_config, last_rows=last_rows)
    elif generator == "itblib":
        features = generate_features_itblib(f_df, gen_config, last_rows=last_rows)
    elif generator == "itbstats":
        features = generate_features_itbstats(f_df, gen_config, last_rows=last_rows)
    elif generator == "highlow":
        features = generate_labels_highlow(f_df, horizon=gen_config.get("horizon"))
    elif generator == "highlow2":
        f_df, features = generate_labels_highlow2(f_df, gen_config)
    elif generator == "topbot":
        top_fracs = [0.01, 0.02, 0.03, 0.04, 0.05]
        f_df, features = generate_labels_topbot(f_df, gen_config.get("columns", "close"), top_fracs, [-x for x in top_fracs])
    elif generator == "topbot2":
        f_df, features = generate_labels_topbot2(f_df, gen_config)
    elif generator == "smoothen":
        f_df, features = generate_smoothen_scores(f_df, gen_config)
    elif generator == "combine":
        f_df, features = generate_combine_scores(f_df, gen_config)
    elif generator == "threshold_rule":
        f_df, features = generate_threshold_rule(f_df, gen_config)
    elif generator == "threshold_rule2":
        f_df, features = generate_threshold_rule2(f_df, gen_config)
    else:
        generator_fn = resolve_generator_name(generator)
        if generator_fn is None:
            raise ValueError(f"Unknown feature generator name or unresolvable: {generator!r}")
        f_df, features = generator_fn(f_df, gen_config, config, model_store)

    f_df = f_df[features]
    fp = fs.get("feature_prefix")
    if fp:
        f_df = f_df.add_prefix(fp + "_")

    new_features = f_df.columns.to_list()
    df = df.drop(columns=list(set(df.columns) & set(new_features)))
    df = df.join(f_df)

    return df, new_features


def get_features_labels_algorithms(fs: dict, config: dict) -> tuple[list[str], list[str], list[dict]]:
    """Resolve a train_feature_sets entry's feature/label/algorithm lists, falling back to the
    config's top-level ``train_features``/``labels``/``algorithms`` when the entry doesn't
    override them.
    """
    fs_config = fs.get("config", {})

    train_features = fs_config.get("columns") or fs_config.get("features") or config.get("train_features", [])
    labels = fs_config.get("labels") or config.get("labels", [])

    algorithms_all = config.get("algorithms")
    algorithm_names = fs_config.get("functions") or fs_config.get("algorithms") or []
    algorithms = []
    for alg in algorithm_names:
        if isinstance(alg, str):
            alg = find_algorithm_by_name(algorithms_all, alg)
        elif not isinstance(alg, dict):
            raise ValueError("Algorithm entry must be either a dict or a name string.")
        algorithms.append(alg)
    if not algorithms:
        algorithms = algorithms_all

    return train_features, labels, algorithms


def _train_one(algo_type: str, df_X: pd.DataFrame, df_y: pd.Series, model_config: dict) -> tuple:
    if algo_type == "lc":
        from common.classifiers.classifier_lc import train_lc

        return train_lc(df_X, df_y, model_config)
    if algo_type == "gb":
        from common.classifiers.classifier_gb import train_gb

        return train_gb(df_X, df_y, model_config)
    if algo_type == "nn":
        from common.classifiers.classifier_nn import train_nn

        return train_nn(df_X, df_y, model_config)
    if algo_type == "svc":
        from common.classifiers.classifier_svc import train_svc

        return train_svc(df_X, df_y, model_config)
    raise ValueError(f"Unknown algorithm type {algo_type!r}.")


def _predict_one(algo_type: str, model_pair: tuple, df_X_test: pd.DataFrame, model_config: dict) -> pd.Series:
    if algo_type == "lc":
        from common.classifiers.classifier_lc import predict_lc

        return predict_lc(model_pair, df_X_test, model_config)
    if algo_type == "gb":
        from common.classifiers.classifier_gb import predict_gb

        return predict_gb(model_pair, df_X_test, model_config)
    if algo_type == "nn":
        from common.classifiers.classifier_nn import predict_nn

        return predict_nn(model_pair, df_X_test, model_config)
    if algo_type == "svc":
        from common.classifiers.classifier_svc import predict_svc

        return predict_svc(model_pair, df_X_test, model_config)
    raise ValueError(f"Unknown algorithm type {algo_type!r}.")


def train_feature_set(df: pd.DataFrame, fs: dict, config: dict) -> tuple[dict[str, tuple], dict[str, object]]:
    """Returns ``(models, calibrators)`` — ``calibrators`` is empty unless a config entry sets
    ``algorithms[].params.calibrate: true`` (v2 item 3, CHANGELOG_V2.md)."""
    train_features, labels, algorithms = get_features_labels_algorithms(fs, config)

    df = df.dropna(subset=train_features).reset_index(drop=True)
    df = df.dropna(subset=labels).reset_index(drop=True)

    models: dict[str, tuple] = {}
    calibrators: dict[str, object] = {}  # v2 item 3 (CHANGELOG_V2.md), empty unless requested
    for label in labels:
        for model_config in algorithms:
            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = f"{label}_{algo_name}"

            train_df = df
            every_nth = model_config.get("params", {}).get("every_nth_row")
            if every_nth:
                train_df = train_df.iloc[::every_nth, :]
            train_length = model_config.get("params", {}).get("length")
            if train_length:
                train_df = train_df.tail(train_length)

            df_X = train_df[train_features]
            df_y = train_df[label]

            print(
                f"Train {score_column_name!r}. Algorithm {algo_name}. Label: {label}. "
                f"Train length {len(df_X)}. Train columns {len(df_X.columns)}"
            )
            model_pair = _train_one(algo_type, df_X, df_y, model_config)
            models[score_column_name] = model_pair

            if model_config.get("params", {}).get("calibrate"):
                from v2.calibration import default_method_for_algo, fit_calibrator

                raw_scores = _predict_one(algo_type, model_pair, df_X, model_config)
                method = model_config.get("params", {}).get("calibrate_method") or default_method_for_algo(algo_type)
                print(f"  Fitting {method!r} calibrator for {score_column_name!r}...")
                calibrators[score_column_name] = fit_calibrator(raw_scores, df_y, method=method)

    return models, calibrators


def predict_feature_set(
    df: pd.DataFrame, fs: dict, config: dict, model_store: ModelStore
) -> tuple[pd.DataFrame, list[str]]:
    train_features, labels, algorithms = get_features_labels_algorithms(fs, config)
    train_df = df[train_features]

    features: list[str] = []
    out_df = pd.DataFrame(index=train_df.index)

    for label in labels:
        for model_config in algorithms:
            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = f"{label}_{algo_name}"

            model_pair = model_store.get_model_pair(score_column_name)
            print(
                f"Predict {score_column_name!r}. Algorithm {algo_name}. Label: {label}. "
                f"Train length {len(train_df)}. Train columns {len(train_df.columns)}"
            )
            scores = _predict_one(algo_type, model_pair, train_df, model_config)

            calibrator = model_store.get_calibrator(score_column_name)  # v2 item 3, None unless trained with it
            if calibrator is not None:
                from v2.calibration import apply_calibrator

                scores = apply_calibrator(calibrator, scores)

            out_df[score_column_name] = scores
            features.append(score_column_name)

    return out_df, features


async def output_feature_set(df: pd.DataFrame, fs: dict, config: dict, model_store: ModelStore) -> None:
    """Dispatch an output/notification/execution generator (used by ``scripts/output.py`` and,
    every tick, by the live service in ``service/server.py``).
    """
    generator = fs.get("generator")
    gen_config = fs.get("config", {})

    if generator == "score_notification_model":
        from outputs.notifier_scores import send_score_notification as generator_fn
    elif generator == "diagram_notification_model":
        from outputs.notifier_diagram import send_diagram as generator_fn
    elif generator == "trader_simulation":
        from outputs.notifier_trades import trader_simulation as generator_fn
    elif generator == "trader_binance":
        # Independent, environment-level kill switch -- deliberately outside the JSONC config
        # file so an accidental config edit alone can never enable real order submission. See
        # the README's "Safety / Guardrails" section and service/server.py's startup check
        # (which additionally requires --i-understand-live-trading-risk before even reaching
        # this point in a live server run).
        if os.environ.get("ITB_ALLOW_LIVE_TRADING") != "1":
            log.warning(
                "output_sets includes 'trader_binance' but ITB_ALLOW_LIVE_TRADING=1 is not set "
                "in the environment. Skipping this output -- no order will be submitted."
            )
            return
        from outputs.trader_binance import trader_binance as generator_fn
    else:
        generator_fn = resolve_generator_name(generator)
        if generator_fn is None:
            raise ValueError(f"Unknown output generator name or unresolvable: {generator!r}")

    if asyncio.iscoroutinefunction(generator_fn):
        await generator_fn(df, gen_config, config, model_store)
    else:
        generator_fn(df, gen_config, config, model_store)
