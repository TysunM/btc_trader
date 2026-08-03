"""Tests for common/model_store.py: label-algo model persistence round-trip, the generic named
model registry, and v2 calibrator persistence (CHANGELOG_V2.md item 3).
"""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common.model_store import ModelStore, resolve_algorithms_for_generator, score_to_label_algo_pair


@pytest.fixture
def store_config(base_config) -> dict:
    base_config["labels"] = ["high_20", "low_20"]
    base_config["algorithms"] = [{"name": "lc", "algo": "lc"}]
    base_config["train_feature_sets"] = [{"generator": "train_features", "config": {}}]
    return base_config


class TestModelPairRoundTrip:
    def test_put_then_get_returns_same_pair(self, store_config):
        store = ModelStore(store_config)
        model = LogisticRegression()
        scaler = StandardScaler()
        store.put_model_pair("high_20_lc", (model, scaler))

        retrieved = store.get_model_pair("high_20_lc")
        assert retrieved[0] is model
        assert retrieved[1] is scaler

    def test_put_creates_files_on_disk(self, store_config):
        store = ModelStore(store_config)
        store.put_model_pair("high_20_lc", (LogisticRegression(), None))
        assert (store.model_path / "high_20_lc.pickle").is_file()
        assert (store.model_path / "high_20_lc.scaler").is_file()

    def test_load_models_reads_declared_label_algo_combinations(self, store_config):
        store = ModelStore(store_config)
        store.put_model_pair("high_20_lc", (LogisticRegression(), StandardScaler()))
        store.put_model_pair("low_20_lc", (LogisticRegression(), StandardScaler()))

        fresh_store = ModelStore(store_config)
        fresh_store.load_models()
        assert set(fresh_store.model_pairs.keys()) == {"high_20_lc", "low_20_lc"}

    def test_load_models_skips_missing_files_without_raising(self, store_config):
        # Nothing has been trained/saved yet -- load_models should not raise, just log and skip.
        store = ModelStore(store_config)
        store.load_models()
        assert store.model_pairs == {}

    def test_get_missing_model_pair_raises_keyerror(self, store_config):
        store = ModelStore(store_config)
        with pytest.raises(KeyError):
            store.get_model_pair("nonexistent_label_algo")


class TestCalibratorPersistence:
    def test_put_then_get_calibrator(self, store_config):
        from sklearn.isotonic import IsotonicRegression

        store = ModelStore(store_config)
        calibrator = IsotonicRegression()
        calibrator.fit([0.1, 0.5, 0.9], [0, 1, 1])
        store.put_calibrator("high_20_lc", calibrator)

        assert store.get_calibrator("high_20_lc") is calibrator
        assert (store.model_path / "high_20_lc.calib").is_file()

    def test_get_calibrator_returns_none_when_absent(self, store_config):
        store = ModelStore(store_config)
        assert store.get_calibrator("never_calibrated") is None

    def test_calibrator_survives_reload_alongside_model(self, store_config):
        from sklearn.isotonic import IsotonicRegression

        store = ModelStore(store_config)
        store.put_model_pair("high_20_lc", (LogisticRegression(), StandardScaler()))
        calibrator = IsotonicRegression()
        calibrator.fit([0.1, 0.5, 0.9], [0, 1, 1])
        store.put_calibrator("high_20_lc", calibrator)

        fresh_store = ModelStore(store_config)
        fresh_store.load_models()
        assert "high_20_lc" in fresh_store.calibrators

    def test_model_without_calibrator_reloads_cleanly(self, store_config):
        # A model trained WITHOUT calibrate=true must not error when reloaded -- get_calibrator
        # should simply return None for it.
        store = ModelStore(store_config)
        store.put_model_pair("low_20_lc", (LogisticRegression(), StandardScaler()))

        fresh_store = ModelStore(store_config)
        fresh_store.load_models()
        assert fresh_store.get_calibrator("low_20_lc") is None


class TestScoreToLabelAlgoPair:
    def test_splits_from_the_right(self):
        # Label names can themselves contain underscores (e.g. "high_20"), so the split must be
        # from the right to correctly separate the trailing algorithm name.
        assert score_to_label_algo_pair("high_20_lc") == ("high_20", "lc")

    def test_handles_multi_underscore_algo_name_incorrectly_by_design(self):
        # rsplit(sep, 1) only ever peels off the LAST underscore-separated token as the algo
        # name -- documenting this as the actual (upstream-matching) behavior.
        assert score_to_label_algo_pair("a_b_c") == ("a_b", "c")


class TestResolveAlgorithmsForGenerator:
    def test_resolves_string_names_against_default_list(self):
        defaults = [{"name": "lc", "algo": "lc"}, {"name": "gb", "algo": "gb"}]
        resolved = resolve_algorithms_for_generator(["gb"], defaults)
        assert resolved == [{"name": "gb", "algo": "gb"}]

    def test_passes_through_dict_entries_unresolved(self):
        custom = {"name": "custom", "algo": "lc"}
        resolved = resolve_algorithms_for_generator([custom], [])
        assert resolved == [custom]

    def test_empty_names_falls_back_to_defaults(self):
        defaults = [{"name": "lc", "algo": "lc"}]
        resolved = resolve_algorithms_for_generator([], defaults)
        assert resolved == defaults
