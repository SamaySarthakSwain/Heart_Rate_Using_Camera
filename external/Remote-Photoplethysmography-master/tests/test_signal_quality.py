"""Unit tests for signal quality evaluator."""

import pytest
from backend.services.signal_quality import SignalQualityEvaluator


class TestSignalQuality:
    """Test suite for signal quality evaluation."""

    def setup_method(self):
        self.evaluator = SignalQualityEvaluator()

    def test_perfect_conditions(self):
        """All good metrics should produce high confidence."""
        result = self.evaluator.evaluate(
            snr=5.0,           # Good SNR
            motion_score=1.0,  # No motion
            brightness=130.0,  # Good lighting
            hr_stability=0.9,  # Stable HR
        )
        assert result["confidence"] > 0.7
        assert result["quality_label"] == "good"

    def test_poor_conditions(self):
        """All bad metrics should produce low confidence."""
        result = self.evaluator.evaluate(
            snr=0.5,           # Poor SNR
            motion_score=0.1,  # Lots of motion
            brightness=20.0,   # Dark
            hr_stability=0.1,  # Unstable
        )
        assert result["confidence"] < 0.4
        assert result["quality_label"] == "poor"

    def test_medium_conditions(self):
        """Mixed metrics should produce fair confidence."""
        result = self.evaluator.evaluate(
            snr=3.0,
            motion_score=0.5,
            brightness=100.0,
            hr_stability=0.5,
        )
        assert 0.3 <= result["confidence"] <= 0.8
        assert result["quality_label"] in ("fair", "good")

    def test_dark_lighting(self):
        """Very dark conditions should have low lighting score."""
        result = self.evaluator.evaluate(
            snr=5.0, motion_score=1.0, brightness=10.0, hr_stability=1.0
        )
        assert result["lighting_score"] < 0.3

    def test_bright_lighting(self):
        """Over-exposed conditions should have lower lighting score."""
        result = self.evaluator.evaluate(
            snr=5.0, motion_score=1.0, brightness=250.0, hr_stability=1.0
        )
        assert result["lighting_score"] < 0.5

    def test_confidence_bounds(self):
        """Confidence should always be in [0, 1]."""
        # Extreme high values
        result = self.evaluator.evaluate(
            snr=100.0, motion_score=1.0, brightness=130.0, hr_stability=1.0
        )
        assert 0.0 <= result["confidence"] <= 1.0

        # Extreme low values
        result = self.evaluator.evaluate(
            snr=-5.0, motion_score=-1.0, brightness=0.0, hr_stability=-1.0
        )
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_scores_present(self):
        """Result should contain all component scores."""
        result = self.evaluator.evaluate(
            snr=3.0, motion_score=0.8, brightness=120.0, hr_stability=0.7
        )
        assert "confidence" in result
        assert "snr_score" in result
        assert "motion_score" in result
        assert "lighting_score" in result
        assert "stability_score" in result
        assert "quality_label" in result
