"""The shared online/offline in-memory state object.

This is the piece that makes ITB's core invariant hold: :meth:`Analyzer.analyze` calls the exact
same :func:`~common.generators.generate_feature_set`/:func:`~common.generators.predict_feature_set`
dispatch functions the offline batch scripts use, so live predictions are computed identically
to how the model was trained — never a second, drifted implementation.

Ported from upstream ITB's ``common/analyzer.py``, adapted to take ``config``/``model_store``
explicitly (already the case upstream — this class never touched the global ``App``) rather
than any change in behavior.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from common.generators import generate_feature_set, predict_feature_set
from common.model_store import ModelStore
from common.utils import (
    append_df_drop_concat,
    get_interval_count_from_start_dt,
    get_start_dt_for_interval_count,
    merge_data_sources,
    notnull_tail_rows,
)

log = logging.getLogger("analyzer")


class Analyzer:
    """In-memory rolling window of trading data (source + derived columns) and its history."""

    def __init__(self, config: dict, model_store: ModelStore):
        self.config = config
        self.model_store = model_store

        # min_window_length: how much history is needed to keep predict_length rows valid given
        # the longest feature lookback. max_window_length: the point at which older rows are
        # trimmed back down to min_window_length (+ a small buffer).
        self.min_window_length = self.config["predict_length"] + self.config["features_horizon"]
        self.max_window_length = self.min_window_length + 15

        # How many tail rows are stale and need (re-)computation. -1 means "recompute
        # everything" (cold start / batch mode). 0 means fully up to date. A positive N is the
        # incremental fast path: only the last N rows need recomputing after an append.
        self.dirty_records = -1

        self.is_train = config.get("train")
        if self.is_train:
            log.warning("Analyzer created with config['train']=True; it is intended for predict-only (server) use.")

        train_features = self.config.get("train_features", [])
        train_features_dtypes = {k: "float64" for k in train_features}

        labels = self.config.get("labels", [])
        labels_dtypes = {k: "float64" for k in labels}

        time_column = self.config["time_column"]
        freq = self.config["freq"]
        all_columns_dtypes = {time_column: "datetime64[ns, UTC]"} | train_features_dtypes
        if self.is_train:
            all_columns_dtypes = all_columns_dtypes | labels_dtypes

        self.df = pd.DataFrame(columns=all_columns_dtypes).astype(all_columns_dtypes)
        self.df = self.df.set_index(time_column, drop=False)
        self.df = self.df.asfreq(freq)

        self.previous_df: pd.DataFrame | None = None  # last 10 rows before the most recent append, for self-validation

    # --- Data state queries ---

    def get_size(self) -> int:
        return len(self.df)

    def get_last_kline(self):
        return self.df.iloc[-1] if len(self.df) > 0 else None

    def get_last_kline_dt(self):
        """Open time of the last kline (also its id)."""
        if len(self.df) > 0:
            return self.df.index[-1]
        freq = self.config["freq"]
        return get_start_dt_for_interval_count(freq, self.min_window_length)

    def get_missing_klines_count(self) -> int:
        last_kline_dt = self.get_last_kline_dt()
        if not last_kline_dt:
            return self.min_window_length
        return get_interval_count_from_start_dt(self.config["freq"], last_kline_dt)

    # --- Data mutation ---

    def append_data(self, dfs: dict[str, pd.DataFrame]) -> None:
        """Merge freshly fetched per-symbol DataFrames onto a common time raster and append,
        overwriting the overlap region with the new values, and update ``dirty_records``.
        """
        data_sources = self.config.get("data_sources", [])
        if len(dfs) != len(data_sources):
            log.warning(f"Retrieved {len(dfs)} symbol(s) but there are {len(data_sources)} configured data source(s).")
        for ds in data_sources:
            ds["df"] = dfs.get(ds.get("folder"))

        time_column = self.config["time_column"]
        freq = self.config["freq"]
        merge_interpolate = self.config.get("merge_interpolate", False)
        df = merge_data_sources(data_sources, time_column, freq, merge_interpolate)

        self.previous_df = self.df.tail(10).copy()

        initial_len = len(self.df)
        appended_len = len(df)
        self.df = append_df_drop_concat(self.df, df)
        result_len = len(self.df)
        overwritten_len = (initial_len + appended_len) - result_len

        if self.dirty_records < 0:
            pass  # already a full-recompute
        elif initial_len == 0:
            self.dirty_records = -1
        else:
            self.dirty_records = appended_len + max(0, self.dirty_records - overwritten_len)
            if self.dirty_records >= result_len:
                self.dirty_records = -1

    def analyze(self) -> None:
        """Recompute derived columns for the dirty tail of ``self.df``: features -> ML
        predictions -> signals, in that order — the same three stages, in the same order,
        as the offline scripts (features.py / predict.py / signals.py).
        """
        symbol = self.config["symbol"]

        if self.dirty_records == 0:
            log.warning("analyze() called with 0 dirty records; nothing to do.")
            return

        last_rows = self.dirty_records if self.dirty_records > 0 else 0

        last_kline_dt = self.get_last_kline_dt()
        log.info(f"Analyze {symbol}. Last kline timestamp: {last_kline_dt}")

        # 1. Features
        feature_sets = self.config.get("feature_sets", [])
        for fs in feature_sets:
            self.df, _ = generate_feature_set(self.df, fs, self.config, self.model_store, last_rows=last_rows)

        # 2. ML predictions
        train_features = self.config["train_features"]
        tail_rows = notnull_tail_rows(self.df[train_features])
        predict_size = tail_rows if not last_rows else min(tail_rows, last_rows)
        predict_features_df = self.df.tail(predict_size)[train_features]

        if predict_features_df.isnull().any().any():
            null_columns = {k: v for k, v in predict_features_df.isnull().any().to_dict().items() if v}
            log.error(f"Null values in predict input. Columns with nulls: {null_columns}")
            return

        train_feature_sets = self.config.get("train_feature_sets", [])
        predict_labels_df = pd.DataFrame(index=predict_features_df.index)
        for fs in train_feature_sets:
            fs_df, _ = predict_feature_set(predict_features_df, fs, self.config, self.model_store)
            predict_labels_df = pd.concat([predict_labels_df, fs_df], axis=1)

        self.df = self.df.combine_first(predict_labels_df)
        self.df.update(predict_labels_df)

        # 3. Signals
        signal_sets = self.config.get("signal_sets", [])
        signal_columns: list[str] = []
        for fs in signal_sets:
            self.df, feats = generate_feature_set(self.df, fs, self.config, self.model_store, last_rows=last_rows)
            signal_columns.extend(feats)

        row = self.get_last_kline()
        scores = ", ".join(
            f"{x}={row[x]:+.3f}" if isinstance(row[x], float) else f"{x}={row[x]}" for x in signal_columns
        )
        log.info(f"Analyze finished. Close: {row['close']:,.2f} Signals: {scores}")

        # Self-check: newly (re)computed overlap rows should be close to what was computed last
        # time. Mismatches are logged, not fatal -- they're a drift/bug signal, not necessarily
        # an error (e.g. a kline can legitimately get revised shortly after it closes).
        check_row_count = 3
        if self.previous_df is not None:
            num_cols = self.previous_df.select_dtypes((float, int)).columns.tolist()
            for r in range(2, min(check_row_count, len(self.df))):
                idx = self.df.index[-r - 1]
                if idx not in self.previous_df.index:
                    continue
                old_row = self.previous_df[num_cols].loc[idx]
                new_row = self.df[num_cols].loc[idx]
                close_mask = np.isclose(old_row, new_row)
                if not np.all(close_mask):
                    log.warning(
                        f"Recomputed row differs from previous computation at {idx}. "
                        f"NEW: {new_row[~close_mask].to_dict()}. OLD: {old_row[~close_mask].to_dict()}"
                    )

        self.dirty_records = 0

        if len(self.df) > self.max_window_length:
            self.df = self.df.tail(self.max_window_length)
