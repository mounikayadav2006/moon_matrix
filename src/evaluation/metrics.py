"""Aggregate evaluation metrics for one full pipeline run."""
from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class PipelineMetrics:
    experiment_name: str = "Baseline"
    num_keypoints_ref: int = 0
    num_keypoints_src: int = 0
    num_candidate_matches: int = 0
    num_good_matches: int = 0
    num_inliers: int = 0
    num_outliers: int = 0
    inlier_ratio: float = 0.0
    spatial_coverage_pct: float = 0.0
    distribution_score: float = 0.0
    registration_error_px: float | None = None   # vs ground truth, if available
    refinement_error_px: float | None = None     # mean sub-pixel displacement
    runtime_sec: float = 0.0
    success: bool = False
    failure_reason: str = ""

    def as_dict(self):
        return {
            "Experiment": self.experiment_name,
            "Keypoints (ref)": self.num_keypoints_ref,
            "Keypoints (src)": self.num_keypoints_src,
            "Candidate Matches": self.num_candidate_matches,
            "Good Matches": self.num_good_matches,
            "Inliers": self.num_inliers,
            "Outliers": self.num_outliers,
            "Inlier Ratio": round(self.inlier_ratio, 4),
            "Spatial Coverage %": round(self.spatial_coverage_pct, 2),
            "Distribution Score": round(self.distribution_score, 3),
            "Registration Error (px)": (
                round(self.registration_error_px, 3) if self.registration_error_px is not None else "N/A"
            ),
            "Refinement Error (px)": (
                round(self.refinement_error_px, 4) if self.refinement_error_px is not None else "N/A"
            ),
            "Runtime (s)": round(self.runtime_sec, 3),
            "Status": "SUCCESS" if self.success else "FAILED",
        }


class Timer:
    """Small context manager to measure wall-clock runtime of a pipeline stage/run."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
