"""
Cross-modal / advanced matching module architecture.

SIH26166 ultimately requires matching across OHRC, TMC-2 and IIRS, which have
very different resolutions, band counts, and radiometry — classic SIFT/BF
matching degrades badly across these modalities.

HONESTY NOTE (see LIMITATIONS.md):
    This prototype does NOT bundle SuperPoint / LightGlue / LoFTR. Those are
    multi-hundred-MB deep learning models with heavy GPU-friendly
    dependencies (torch, kornia, checkpoints) that are not reliable to
    install/run inside a lightweight hackathon demo environment offline.

    Rather than fake cross-modal results, this module:
      1. Runs the SAME baseline SIFT pipeline across sensor pairs (works
         reasonably for same-sensor or visually-similar pairs like OHRC<->TMC-2).
      2. Explicitly reports when a pair is a known "hard" cross-modal case
         (e.g. involving IIRS) so the UI can label it as an "advanced /
         research mode" instead of silently returning a poor-quality result
         framed as success.
      3. Exposes a clean AdvancedMatcher interface that a future integration
         of SuperPoint/LightGlue/LoFTR can implement without touching the
         rest of the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

# Sensor pairs known to be difficult for classical hand-crafted descriptors
HARD_CROSS_MODAL_PAIRS = {
    ("OHRC", "IIRS"), ("IIRS", "OHRC"),
    ("TMC-2", "IIRS"), ("IIRS", "TMC-2"),
    ("IIRS", "IIRS"),  # IIRS is hyperspectral; even self-pairs need band selection logic not yet implemented
}


@dataclass
class SensorPairInfo:
    sensor_a: str
    sensor_b: str
    is_hard_cross_modal: bool
    recommended_mode: str   # "baseline" or "advanced_required"
    note: str


def classify_sensor_pair(sensor_a: str, sensor_b: str) -> SensorPairInfo:
    pair = (sensor_a, sensor_b)
    is_hard = pair in HARD_CROSS_MODAL_PAIRS

    if is_hard:
        return SensorPairInfo(
            sensor_a=sensor_a, sensor_b=sensor_b, is_hard_cross_modal=True,
            recommended_mode="advanced_required",
            note=(f"{sensor_a} \u2194 {sensor_b} is a known hard cross-modal pairing. "
                  "The classical SIFT baseline is run for demonstration purposes, but "
                  "results should be treated as an 'advanced/research mode' preview, "
                  "not a validated correspondence."),
        )

    return SensorPairInfo(
        sensor_a=sensor_a, sensor_b=sensor_b, is_hard_cross_modal=False,
        recommended_mode="baseline",
        note=f"{sensor_a} \u2194 {sensor_b} is supported by the baseline SIFT pipeline.",
    )


class AdvancedMatcher(ABC):
    """Interface for future learned cross-modal matchers (SuperPoint/LightGlue/LoFTR)."""

    name: str = "AdvancedMatcher"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True only if the required model/weights/deps are actually installed."""
        raise NotImplementedError

    @abstractmethod
    def match(self, reference, source):
        raise NotImplementedError


class NotImplementedAdvancedMatcher(AdvancedMatcher):
    """Placeholder used when no advanced matcher is installed. Never fakes a result."""

    name = "None (baseline SIFT only)"

    def is_available(self) -> bool:
        return False

    def match(self, reference, source):
        raise NotImplementedError(
            "No advanced cross-modal matcher (SuperPoint/LightGlue/LoFTR) is installed in "
            "this prototype. See ARCHITECTURE.md and LIMITATIONS.md for the integration plan."
        )


def get_advanced_matcher() -> AdvancedMatcher:
    """
    Factory for the advanced matcher. Currently always returns the honest
    'not implemented' placeholder. A future integration can detect installed
    packages (e.g. `import lightglue`) here and return a real implementation.
    """
    return NotImplementedAdvancedMatcher()
