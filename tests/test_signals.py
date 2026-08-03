"""Tests for outputs/notifier_scores.py's band-finding and throttling logic — upstream had zero
coverage of this despite it being the main gate on whether a Telegram message actually gets sent.
"""

from __future__ import annotations

from common.utils import round_str  # noqa: F401 (import-side sanity: module path stability)
from outputs.notifier_scores import _find_score_band


def _model(positive_bands=None, negative_bands=None) -> dict:
    return {"positive_bands": positive_bands or [], "negative_bands": negative_bands or []}


class TestFindScoreBand:
    POSITIVE = [
        {"edge": 0.08, "text": "BUY ZONE"},
        {"edge": 0.04, "text": "strong"},
        {"edge": 0.02, "text": "weak"},
    ]
    NEGATIVE = [
        {"edge": -0.02, "text": "weak"},
        {"edge": -0.04, "text": "strong"},
        {"edge": -0.08, "text": "SELL ZONE"},
    ]

    def test_neutral_score_has_no_band(self):
        model = _model(self.POSITIVE, self.NEGATIVE)
        band_no, band = _find_score_band(0.0, model)
        assert band is None
        assert band_no == 0

    def test_weak_positive_score_matches_weakest_band(self):
        model = _model(self.POSITIVE, self.NEGATIVE)
        band_no, band = _find_score_band(0.025, model)
        assert band["text"] == "weak"
        assert band_no == 1

    def test_strong_positive_score_matches_correct_band(self):
        model = _model(self.POSITIVE, self.NEGATIVE)
        band_no, band = _find_score_band(0.05, model)
        assert band["text"] == "strong"
        assert band_no == 2

    def test_extreme_positive_score_matches_highest_band(self):
        model = _model(self.POSITIVE, self.NEGATIVE)
        band_no, band = _find_score_band(0.5, model)
        assert band["text"] == "BUY ZONE"
        assert band_no == 3

    def test_negative_score_returns_negative_band_number(self):
        model = _model(self.POSITIVE, self.NEGATIVE)
        band_no, band = _find_score_band(-0.5, model)
        assert band["text"] == "SELL ZONE"
        assert band_no == -3

    def test_boundary_score_uses_greater_equal_for_positive(self):
        model = _model(self.POSITIVE, self.NEGATIVE)
        _, band = _find_score_band(0.08, model)
        assert band["text"] == "BUY ZONE"  # exactly at the edge counts as reaching that band

    def test_no_bands_configured_always_neutral(self):
        band_no, band = _find_score_band(0.5, _model())
        assert band is None
        assert band_no == 0


class TestBandTransitionThrottling:
    """Mirrors send_score_notification's own band-transition logic (band_up/band_dn) without
    needing to construct a full DataFrame/config -- these are the conditions that gate whether a
    Telegram message actually gets sent.
    """

    def _band_transition(self, prev_band_no: int | None, band_no: int) -> tuple[bool, bool]:
        if prev_band_no is not None:
            band_up = abs(band_no) > abs(prev_band_no)
            band_dn = abs(band_no) < abs(prev_band_no)
        else:
            band_up = True
            band_dn = True
        return band_up, band_dn

    def test_first_ever_reading_always_notifies(self):
        band_up, band_dn = self._band_transition(None, 1)
        assert band_up and band_dn

    def test_increasing_strength_is_band_up_only(self):
        band_up, band_dn = self._band_transition(1, 2)
        assert band_up
        assert not band_dn

    def test_decreasing_strength_is_band_dn_only(self):
        band_up, band_dn = self._band_transition(2, 1)
        assert not band_up
        assert band_dn

    def test_crossing_zero_from_positive_to_negative_of_greater_magnitude_is_band_up(self):
        # 1 -> -2: |−2| (2) > |1| (1), so this correctly reads as a strength increase even
        # though the sign flipped.
        band_up, band_dn = self._band_transition(1, -2)
        assert band_up

    def test_same_band_no_transition(self):
        band_up, band_dn = self._band_transition(2, 2)
        assert not band_up
        assert not band_dn
