"""Tests for v2/purge_embargo.py and its integration into scripts/predict_rolling.py's
walk-forward loop. See v2/purge_embargo.py's docstring for why this is an optional
conservatism margin rather than a fix for a demonstrated leakage bug.
"""

from __future__ import annotations

from v2.purge_embargo import purged_train_end


class TestPurgedTrainEnd:
    def test_zero_embargo_matches_upstream_formula_exactly(self):
        # Upstream: train_end = predict_start - label_horizon - 1
        predict_start, label_horizon = 1000, 24
        assert purged_train_end(predict_start, label_horizon, embargo=0) == predict_start - label_horizon - 1

    def test_embargo_pulls_train_end_further_back(self):
        predict_start, label_horizon, embargo = 1000, 24, 48
        result = purged_train_end(predict_start, label_horizon, embargo)
        assert result == 1000 - 24 - 1 - 48
        assert result < purged_train_end(predict_start, label_horizon, embargo=0)

    def test_last_training_rows_label_horizon_never_reaches_predict_start(self):
        # The core correctness property: the LAST training row's label (which looks
        # label_horizon rows forward) must never look as far as predict_start.
        predict_start, label_horizon = 500, 10
        for embargo in (0, 5, 20):
            train_end = purged_train_end(predict_start, label_horizon, embargo)
            last_training_row = train_end - 1  # train_df = df.iloc[train_start:train_end], exclusive
            label_reaches = last_training_row + label_horizon
            assert label_reaches < predict_start

    def test_default_embargo_is_zero_reproducing_v1_baseline(self):
        # Calling without the embargo kwarg at all must match upstream's own gap.
        assert purged_train_end(200, 5) == purged_train_end(200, 5, embargo=0)


class TestRollingLoopIntegration:
    """Verifies the train_end computation actually used inside predict_rolling.py's loop body
    (same formula, exercised the way the script exercises it) — a change to the formula's
    wiring (not just the standalone function) would be caught here.
    """

    def test_predict_rolling_module_uses_purged_train_end(self):
        import inspect

        import scripts.predict_rolling as pr

        # main is wrapped by @click.command(); the original function lives on .callback.
        source = inspect.getsource(pr.main.callback)
        assert "purged_train_end(predict_start, label_horizon, embargo)" in source

    def test_embargo_read_from_rolling_predict_config_defaults_zero(self):
        import inspect

        import scripts.predict_rolling as pr

        source = inspect.getsource(pr.main.callback)
        assert 'rp_config.get("embargo", 0)' in source
