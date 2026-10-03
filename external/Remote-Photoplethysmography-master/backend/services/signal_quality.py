"""Signal quality evaluation combining multiple quality metrics."""

import numpy as np


class SignalQualityEvaluator:
    """Evaluates overall signal quality for confidence scoring.

    Combines: SNR, motion score, lighting score, signal stability.
    """

    def __init__(
        self,
        snr_weight: float = 0.4,
        motion_weight: float = 0.2,
        lighting_weight: float = 0.2,
        stability_weight: float = 0.2,
        min_brightness: float = 50.0,
        max_brightness: float = 220.0,
    ):
        """Initialize quality evaluator.

        Args:
            snr_weight: Weight for SNR component.
            motion_weight: Weight for motion component.
            lighting_weight: Weight for lighting component.
            stability_weight: Weight for stability component.
            min_brightness: Minimum acceptable brightness (0-255).
            max_brightness: Maximum acceptable brightness (0-255).
        """
        self.snr_weight = snr_weight
        self.motion_weight = motion_weight
        self.lighting_weight = lighting_weight
        self.stability_weight = stability_weight
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def evaluate(
        self,
        snr: float,
        motion_score: float,
        brightness: float,
        hr_stability: float,
    ) -> dict:
        """Compute overall signal quality and confidence score.

        Args:
            snr: Signal-to-noise ratio (linear scale).
            motion_score: Motion quality score (0-1, higher = less motion).
            brightness: Mean ROI brightness (0-255).
            hr_stability: HR stability score (0-1, higher = more stable).

        Returns:
            Dict with 'confidence', 'snr_score', 'motion_score',
            'lighting_score', 'stability_score', 'quality_label'.
        """
        # Normalize SNR to 0-1
        snr_score = min(1.0, max(0.0, (snr - 1.0) / 4.0))

        # Lighting score based on brightness
        lighting_score = self._compute_lighting_score(brightness)

        # Clamp input scores
        motion_score = min(1.0, max(0.0, motion_score))
        stability_score = min(1.0, max(0.0, hr_stability))

        # Weighted combination
        confidence = (
            self.snr_weight * snr_score
            + self.motion_weight * motion_score
            + self.lighting_weight * lighting_score
            + self.stability_weight * stability_score
        )
        confidence = min(1.0, max(0.0, confidence))

        # Quality label
        if confidence >= 0.7:
            quality_label = "good"
        elif confidence >= 0.4:
            quality_label = "fair"
        else:
            quality_label = "poor"

        return {
            "confidence": float(confidence),
            "snr_score": float(snr_score),
            "motion_score": float(motion_score),
            "lighting_score": float(lighting_score),
            "stability_score": float(stability_score),
            "quality_label": quality_label,
        }

    def _compute_lighting_score(self, brightness: float) -> float:
        """Compute lighting quality score from brightness.

        Args:
            brightness: Mean pixel intensity (0-255).

        Returns:
            Lighting score (0-1).
        """
        if brightness < self.min_brightness:
            return max(0.0, brightness / self.min_brightness)
        elif brightness > self.max_brightness:
            return max(0.0, 1.0 - (brightness - self.max_brightness) / (255 - self.max_brightness))
        else:
            # In ideal range
            return 1.0
