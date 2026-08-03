"""Persistent store for trained models.

Two conventions, both ported from upstream ITB:
1. "label-algo" pairs — the convention used by the offline pipeline scripts. A model trained to
   predict label ``high_30`` with algorithm ``lc`` is stored as ``MODELS/high_30_lc.pickle`` +
   ``.scaler``, keyed by the score column name ``<label>_<algo>``.
2. A generic named registry (``config["model_registry"]``) for arbitrary persisted objects
   (e.g. a feature generator's auto-discovered thresholds), format inferred from file extension.

One de-duplication vs. upstream: ``find_algorithm_by_name`` was defined identically in both
``common/utils.py`` and here upstream (copy-paste); this port keeps a single definition in
``common/utils.py`` and imports it.
"""

from __future__ import annotations

import itertools
import json
import logging
import pickle
from pathlib import Path

from joblib import dump, load

from common.utils import find_algorithm_by_name

log = logging.getLogger("model_store")

label_algo_separator = "_"


class ModelStore:
    def __init__(self, config: dict):
        self.config = config

        symbol = config["symbol"]
        data_path = Path(config["data_folder"]) / symbol
        model_path = Path(config["model_folder"])
        if not model_path.is_absolute():
            model_path = data_path / model_path
        self.model_path = model_path.resolve()

        self.model_registry = config.get("model_registry", [])

        self.model_pairs: dict[str, tuple] = {}
        self.models: dict[str, object] = {}
        # v2 item 3 (CHANGELOG_V2.md): optional per-score-column calibrator, absent unless a
        # model was trained with algorithms[].params.calibrate=true.
        self.calibrators: dict[str, object] = {}

    def load_models(self) -> None:
        """Load all models from persistent store into memory."""
        self.model_pairs = self._load_models_for_generators()
        self.calibrators = self._load_calibrators_for_generators()

        for model_entry in self.model_registry:
            model_name = model_entry.get("name")
            model_path = self.model_path / model_entry.get("file")
            ext = model_path.suffix.lower()
            try:
                if ext == ".json":
                    with open(model_path) as f:
                        model_object = json.load(f)
                elif ext in (".txt", ".csv"):
                    model_object = model_path.read_text()
                elif ext in (".pickle", ".scaler"):
                    model_object = load(model_path)
                else:
                    with open(model_path, "rb") as f:
                        model_object = pickle.load(f)
            except Exception:
                model_object = None
            self.models[model_name] = model_object

    def put_model(self, name: str, model) -> None:
        model_entry = next((x for x in self.model_registry if x.get("name") == name), None)
        if not model_entry:
            raise ValueError(f"Model {name!r} is not declared in config['model_registry'].")

        model_path = self.model_path / model_entry.get("file")
        ext = model_path.suffix.lower()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if ext == ".json":
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(model, f, ensure_ascii=False, indent=4)
        elif ext in (".txt", ".csv"):
            model_path.write_text(model)
        elif ext in (".pickle", ".scaler"):
            dump(model, model_path)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        self.models[name] = model

    def get_model(self, name: str):
        return self.models.get(name)

    def get_all_model_pairs(self) -> dict[str, tuple]:
        return self.model_pairs

    def get_model_pair(self, column_name: str) -> tuple:
        return self.model_pairs[column_name]

    def put_model_pair(self, column_name: str, model_pair: tuple) -> None:
        self.model_pairs[column_name] = model_pair
        self._save_label_algo_model_pair_to_file(column_name, model_pair)

    def get_calibrator(self, column_name: str):
        """Return the fitted calibrator for a score column, or None if it was never calibrated
        (v2 item 3, CHANGELOG_V2.md — optional, off by default)."""
        return self.calibrators.get(column_name)

    def put_calibrator(self, column_name: str, calibrator) -> None:
        self.model_path.mkdir(parents=True, exist_ok=True)
        dump(calibrator, (self.model_path / column_name).with_suffix(".calib"))
        self.calibrators[column_name] = calibrator

    def _load_calibrators_for_generators(self) -> dict[str, object]:
        calibrators = {}
        for score_column_name in self.model_pairs:
            calib_path = (self.model_path / score_column_name).with_suffix(".calib")
            if calib_path.is_file():
                try:
                    calibrators[score_column_name] = load(calib_path)
                except Exception:
                    log.error(f"Cannot load calibrator for {score_column_name!r} from {calib_path}. Skipping.")
        return calibrators

    # --- label-algo convention ---

    def _load_models_for_generators(self) -> dict[str, tuple]:
        labels_default = self.config.get("labels", [])
        algorithms_default = self.config.get("algorithms")

        train_feature_sets = self.config.get("train_feature_sets", [])
        models: dict[str, tuple] = {}
        for fs in train_feature_sets:
            labels = fs.get("config", {}).get("labels", []) or labels_default

            algorithm_names = fs.get("config", {}).get("functions", []) or fs.get("config", {}).get("algorithms", [])
            algorithms = resolve_algorithms_for_generator(algorithm_names, algorithms_default)

            models.update(self._load_all_label_algo_model_pairs(labels, algorithms))
        return models

    def _load_all_label_algo_model_pairs(self, labels: list[str], algorithms: list[dict]) -> dict[str, tuple]:
        models = {}
        for label, algorithm in itertools.product(labels, algorithms):
            score_column_name = f"{label}{label_algo_separator}{algorithm['name']}"
            try:
                models[score_column_name] = self._load_label_algo_model_pair_from_file(score_column_name)
            except Exception:
                log.error(f"Cannot load model {score_column_name!r} from {self.model_path}. Skipping.")
        return models

    def _load_label_algo_model_pair_from_file(self, score_column_name: str) -> tuple:
        scaler = load((self.model_path / score_column_name).with_suffix(".scaler"))
        model = load((self.model_path / score_column_name).with_suffix(".pickle"))
        return model, scaler

    def _save_label_algo_model_pair_to_file(self, column_name: str, model_pair: tuple) -> None:
        self.model_path.mkdir(parents=True, exist_ok=True)
        model, scaler = model_pair
        dump(scaler, (self.model_path / column_name).with_suffix(".scaler"))
        dump(model, (self.model_path / column_name).with_suffix(".pickle"))


def resolve_algorithms_for_generator(algorithm_names: list, algorithms_default: list[dict]) -> list[dict]:
    algorithms = []
    for alg in algorithm_names:
        if isinstance(alg, str):
            alg = find_algorithm_by_name(algorithms_default, alg)
        elif not isinstance(alg, dict):
            raise ValueError("Algorithm entry must be either a dict or a name string.")
        algorithms.append(alg)
    return algorithms or algorithms_default


def score_to_label_algo_pair(score_column_name: str) -> tuple[str, str]:
    label_name, algo_name = score_column_name.rsplit(label_algo_separator, 1)
    return label_name, algo_name
