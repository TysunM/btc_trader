"""Score calibration (v2 quant pass, item 3 — see CHANGELOG_V2.md).

Raw scores from different algorithm families aren't comparable probabilities: a LightGBM margin,
a logistic-regression ``predict_proba`` output, and a Keras sigmoid output don't mean the same
thing at the same numeric value, yet ``signals.py``'s ``combine`` generator subtracts two raw
scores directly (e.g. ``high_30_lc - low_30_lc``).

Design note: sklearn's ``CalibratedClassifierCV`` requires an sklearn-compatible estimator
(``fit``/``predict_proba``), which none of this project's four classifier modules uniformly
provide — ``classifier_gb.py`` wraps a raw LightGBM ``Booster``, ``classifier_nn.py`` wraps a
Keras model, neither of which implements that interface. Rather than special-case each family,
this module calibrates the *output scores* directly: fit a monotonic mapping from
(raw score -> true label) once after training, on the same data the model was just trained on,
and apply it to every future raw score at predict time. This works uniformly across all four
algorithm families since it never touches the underlying model object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_METHODS = ("isotonic", "sigmoid")


class _SigmoidCalibrator:
    """Platt scaling (a 1-feature logistic regression from raw score -> P(label=1)) via the
    public sklearn API, rather than importing sklearn.calibration's private
    ``_SigmoidCalibration`` — same underlying idea, but not exposed to breakage if that private
    class is renamed or removed in a future sklearn release.
    """

    def __init__(self):
        self._model = LogisticRegression()

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "_SigmoidCalibrator":
        self._model.fit(scores.reshape(-1, 1), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(scores.reshape(-1, 1))[:, 1]


def default_method_for_algo(algo_type: str) -> str:
    """Isotonic calibration is nonparametric and needs a reasonably large, clean sample to avoid
    overfitting the calibration curve itself — a good fit for gb/nn, which are typically trained
    on the largest data volumes in this pipeline. Sigmoid (Platt scaling) is a 2-parameter fit
    that degrades more gracefully on the smaller effective sample sizes lc/svc are often used
    with (svc in particular is frequently configured with a capped `params.length`).
    """
    return "isotonic" if algo_type in ("gb", "nn") else "sigmoid"


def fit_calibrator(raw_scores: pd.Series, y_true: pd.Series, method: str = "isotonic"):
    """Fit a calibrator mapping raw model scores to calibrated [0,1] probabilities.

    Drops rows where either input is NaN (matches the NaN-safe convention every classifier
    module in this project already follows for predict_X()).
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown calibration method {method!r}. Available: {list(_METHODS)}")

    paired = pd.DataFrame({"score": raw_scores, "y": y_true}).dropna()
    if len(paired) < 10:
        raise ValueError(f"Need at least 10 non-NaN (score, label) pairs to fit a calibrator, got {len(paired)}.")

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    else:
        calibrator = _SigmoidCalibrator()
    calibrator.fit(paired["score"].values, paired["y"].values.astype(float))

    return calibrator


def apply_calibrator(calibrator, raw_scores: pd.Series) -> pd.Series:
    """Apply a fitted calibrator to a (possibly NaN-containing) raw score series, preserving
    the input's index and NaN positions exactly — matching the NaN-safe convention used
    throughout this project's classifier predict_X() functions.
    """
    nonnan = raw_scores.dropna()
    if len(nonnan) == 0:
        return raw_scores.copy()

    calibrated_values = calibrator.predict(nonnan.values)
    out = pd.Series(index=raw_scores.index, dtype=float)
    out.loc[nonnan.index] = np.clip(calibrated_values, 0.0, 1.0)
    return out
