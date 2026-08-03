"""Tests for common/generators.py — the plugin dispatch mechanism that makes the offline/online
feature-parity invariant hold. Covers: every built-in generator conforms to the expected
signature/return shape, the custom-generator (module:function) resolution path works, and
train/predict round-trip through the real classifier dispatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.generators import generate_feature_set, get_features_labels_algorithms, predict_feature_set, train_feature_set
from common.model_store import ModelStore


@pytest.fixture
def model_store(base_config) -> ModelStore:
    return ModelStore(base_config)


class TestGenerateFeatureSetBuiltins:
    def test_talib_generator_produces_expected_columns(self, ohlcv_df, base_config, model_store):
        fs = {"column_prefix": "", "generator": "talib", "feature_prefix": "", "config": {"columns": ["close"], "functions": ["SMA"], "windows": [5, 10]}}
        df, features = generate_feature_set(ohlcv_df.copy(), fs, base_config, model_store)
        assert features == ["close_SMA_5", "close_SMA_10"]
        assert set(features).issubset(df.columns)
        # SMA_5 should be non-NaN well before the end of a 200-row series
        assert df["close_SMA_5"].iloc[-1] == pytest.approx(ohlcv_df["close"].iloc[-5:].mean())

    def test_highlow2_generator_produces_boolean_label(self, ohlcv_df, base_config, model_store):
        fs = {
            "column_prefix": "", "generator": "highlow2", "feature_prefix": "",
            "config": {"columns": ["close", "high", "low"], "function": "high", "thresholds": [2.0], "tolerance": 0.2, "horizon": 10, "names": ["high_20"]},
        }
        df, features = generate_feature_set(ohlcv_df.copy(), fs, base_config, model_store)
        assert features == ["high_20"]
        assert df["high_20"].dtype == bool

    def test_combine_and_threshold_rule_chain(self, base_config, model_store):
        df = pd.DataFrame({"buy_score": [0.8, 0.2, 0.5], "sell_score": [0.1, 0.6, 0.5]})
        combine_fs = {"column_prefix": "", "generator": "combine", "feature_prefix": "", "config": {"columns": ["buy_score", "sell_score"], "names": "trade_score", "combine": "difference"}}
        df, feats1 = generate_feature_set(df, combine_fs, base_config, model_store)
        assert feats1 == ["trade_score"]
        assert list(df["trade_score"]) == pytest.approx([0.7, -0.4, 0.0])

        threshold_fs = {"column_prefix": "", "generator": "threshold_rule", "feature_prefix": "", "config": {"columns": "trade_score", "names": ["buy_signal_column", "sell_signal_column"], "parameters": {"buy_signal_threshold": 0.5, "sell_signal_threshold": -0.3}}}
        df, feats2 = generate_feature_set(df, threshold_fs, base_config, model_store)
        assert list(df["buy_signal_column"]) == [True, False, False]
        assert list(df["sell_signal_column"]) == [False, True, False]

    def test_feature_prefix_is_applied(self, ohlcv_df, base_config, model_store):
        fs = {"column_prefix": "", "generator": "talib", "feature_prefix": "myprefix", "config": {"columns": ["close"], "functions": ["SMA"], "windows": [5]}}
        df, features = generate_feature_set(ohlcv_df.copy(), fs, base_config, model_store)
        assert features == ["myprefix_close_SMA_5"]

    def test_regenerating_a_feature_set_overwrites_not_duplicates(self, ohlcv_df, base_config, model_store):
        fs = {"column_prefix": "", "generator": "talib", "feature_prefix": "", "config": {"columns": ["close"], "functions": ["SMA"], "windows": [5]}}
        df, _ = generate_feature_set(ohlcv_df.copy(), fs, base_config, model_store)
        n_cols_once = len(df.columns)
        df, _ = generate_feature_set(df, fs, base_config, model_store)
        assert len(df.columns) == n_cols_once  # not doubled

    def test_not_yet_ported_generator_raises_clear_error(self, ohlcv_df, base_config, model_store):
        fs = {"column_prefix": "", "generator": "depth", "feature_prefix": "", "config": {}}
        with pytest.raises(NotImplementedError, match="depth"):
            generate_feature_set(ohlcv_df.copy(), fs, base_config, model_store)

    def test_unknown_generator_raises_value_error(self, ohlcv_df, base_config, model_store):
        fs = {"column_prefix": "", "generator": "totally_unknown_generator_xyz", "feature_prefix": "", "config": {}}
        with pytest.raises(ValueError, match="Unknown feature generator"):
            generate_feature_set(ohlcv_df.copy(), fs, base_config, model_store)


class TestColumnPrefixSelection:
    def test_column_prefix_strips_and_reapplies(self, base_config, model_store):
        df = pd.DataFrame({"src_close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        fs = {"column_prefix": "src", "generator": "talib", "feature_prefix": "", "config": {"columns": ["close"], "functions": ["SMA"], "windows": [2]}}
        df, features = generate_feature_set(df, fs, base_config, model_store)
        assert "close_SMA_2" in df.columns
        assert "src_close" in df.columns  # original untouched


class TestCustomGeneratorPlugin:
    def test_my_feature_example_resolves_via_module_colon_function(self, base_config, model_store):
        df = pd.DataFrame({"close": [10.0, 20.0, 30.0]})
        fs = {
            "column_prefix": "", "generator": "common.my_feature_example:my_feature_example", "feature_prefix": "",
            "config": {"columns": "close", "function": "add", "parameter": 5.0, "names": "close_add"},
        }
        df, features = generate_feature_set(df, fs, base_config, model_store)
        assert features == ["close_add"]
        assert list(df["close_add"]) == [15.0, 25.0, 35.0]

    def test_ensemble_resolves_via_plugin_mechanism(self, base_config, model_store):
        df = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]})
        fs = {"column_prefix": "", "generator": "v2.ensemble:generate_ensemble_score", "feature_prefix": "", "config": {"columns": ["a", "b"], "names": "avg", "method": "mean"}}
        df, features = generate_feature_set(df, fs, base_config, model_store)
        assert features == ["avg"]
        assert list(df["avg"]) == pytest.approx([0.2, 0.3])


class TestGetFeaturesLabelsAlgorithms:
    def test_falls_back_to_config_top_level(self, base_config):
        base_config["train_features"] = ["f1", "f2"]
        base_config["labels"] = ["l1"]
        base_config["algorithms"] = [{"name": "a", "algo": "lc"}]
        fs = {"generator": "train_features", "config": {}}
        features, labels, algorithms = get_features_labels_algorithms(fs, base_config)
        assert features == ["f1", "f2"]
        assert labels == ["l1"]
        assert algorithms == [{"name": "a", "algo": "lc"}]

    def test_entry_level_override_wins(self, base_config):
        base_config["train_features"] = ["f1"]
        fs = {"generator": "train_features", "config": {"columns": ["override_feature"]}}
        features, _, _ = get_features_labels_algorithms(fs, base_config)
        assert features == ["override_feature"]


class TestTrainPredictRoundTrip:
    def test_lc_trains_and_predicts_through_real_dispatch(self, base_config):
        n = 300
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "label": rng.integers(0, 2, size=n),
        })
        base_config["train_features"] = ["f1", "f2"]
        base_config["labels"] = ["label"]
        base_config["algorithms"] = [{"name": "lc", "algo": "lc", "params": {"is_scale": True}, "train": {"max_iter": 200}}]

        fs = {"generator": "train_features", "config": {}}
        models, calibrators = train_feature_set(df, fs, base_config)
        assert "label_lc" in models
        assert calibrators == {}  # calibrate not requested

        model_store = ModelStore(base_config)
        model_store.put_model_pair("label_lc", models["label_lc"])

        out_df, features = predict_feature_set(df, fs, base_config, model_store)
        assert features == ["label_lc"]
        assert out_df["label_lc"].between(0, 1).all()

    def test_calibrated_model_persists_and_applies_on_predict(self, base_config):
        n = 300
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "f1": rng.normal(size=n),
            "label": rng.integers(0, 2, size=n),
        })
        base_config["train_features"] = ["f1"]
        base_config["labels"] = ["label"]
        base_config["algorithms"] = [{"name": "lc", "algo": "lc", "params": {"is_scale": True, "calibrate": True}, "train": {"max_iter": 200}}]

        fs = {"generator": "train_features", "config": {}}
        base_config["train_feature_sets"] = [fs]
        models, calibrators = train_feature_set(df, fs, base_config)
        assert "label_lc" in calibrators

        model_store = ModelStore(base_config)
        model_store.put_model_pair("label_lc", models["label_lc"])
        model_store.put_calibrator("label_lc", calibrators["label_lc"])

        reloaded = ModelStore(base_config)
        reloaded.load_models()
        assert "label_lc" in reloaded.calibrators

        out_df, _ = predict_feature_set(df, fs, base_config, reloaded)
        assert out_df["label_lc"].between(0, 1).all()
