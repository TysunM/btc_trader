"""Configurable embargo buffer for walk-forward validation (v2 quant pass, item 2 — see
CHANGELOG_V2.md).

Honest framing, arrived at after actually working through the index arithmetic rather than
assuming the textbook concern applied unmodified: upstream ITB's ``scripts/predict_rolling.py``
already computes ``train_end = predict_start - label_horizon - 1``, which is *sufficient* to
prevent label lookahead across the train/predict boundary for this specific loop design — it's a
strictly sequential, single-direction walk-forward (never revisits earlier windows out of order
the way k-fold cross-validation does), so the classic purged-CV leakage scenario (a test fold's
neighboring train folds peeking at its labels) doesn't actually arise here the way it would in a
shuffled/randomized CV split. The last training row's label horizon reaches at most to
``predict_start - 2`` — still strictly before the predict window.

So this isn't a bug fix. It's an optional extra conservatism knob: an ``embargo`` of N rows
pulls the training cutoff back N rows further than the strict minimum, which is standard
practice in the purged-CV literature as a safety margin against model-estimation uncertainty or
a custom label generator with different/fatter-tailed lookahead characteristics than the
``highlow2`` generator this project ships with. Defaults to 0 (v1 baseline, unchanged).
"""

from __future__ import annotations


def purged_train_end(predict_start: int, label_horizon: int, embargo: int = 0) -> int:
    """Exclusive end index for training data before a predict window starting at
    ``predict_start``. ``embargo=0`` reproduces upstream's existing (already-correct)
    ``predict_start - label_horizon - 1`` gap exactly.
    """
    return predict_start - label_horizon - 1 - embargo
