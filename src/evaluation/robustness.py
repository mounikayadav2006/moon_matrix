"""
Scale and illumination robustness experiments.

Runs the SAME baseline pipeline (preprocessing -> SIFT -> matching -> RANSAC
-> spatial analysis) against systematically perturbed versions of the source
image, so performance change vs the perturbation can be reported with
genuinely computed numbers rather than assumptions.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.features.sift import detect_sift_features
from src.matching.matcher import match_features, InsufficientFeaturesError
from src.geometry.ransac import run_ransac
from src.spatial.grid import analyze_spatial_distribution
from src.evaluation.metrics import PipelineMetrics, Timer
from src.utils.sample_data import apply_illumination


SCALE_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]
ILLUMINATION_PRESETS = {
    "Brighter (+40)": {"brightness_delta": 40, "gamma": 1.0},
    "Darker (-40)": {"brightness_delta": -40, "gamma": 1.0},
    "Low contrast (gamma 1.8)": {"brightness_delta": 0, "gamma": 1.8},
    "High contrast (gamma 0.6)": {"brightness_delta": 0, "gamma": 0.6},
}


def _run_baseline_pipeline(reference: np.ndarray, source: np.ndarray, experiment_name: str,
                            transform_type: str = "homography", grid_size: int = 4,
                            ratio: float = 0.75) -> PipelineMetrics:
    m = PipelineMetrics(experiment_name=experiment_name)
    with Timer() as t:
        ref_feat = detect_sift_features(reference)
        src_feat = detect_sift_features(source)
        m.num_keypoints_ref = ref_feat.num_keypoints
        m.num_keypoints_src = src_feat.num_keypoints

        try:
            match_res = match_features(ref_feat.descriptors, src_feat.descriptors, ratio=ratio)
        except InsufficientFeaturesError as e:
            m.success = False
            m.failure_reason = str(e)
            m.runtime_sec = 0.0
            return m

        m.num_candidate_matches = match_res.num_candidates
        m.num_good_matches = match_res.num_good

        ransac_res = run_ransac(ref_feat.keypoints, src_feat.keypoints, match_res.good_matches,
                                 transform_type=transform_type)
        m.num_inliers = ransac_res.num_inliers
        m.num_outliers = ransac_res.num_outliers
        m.inlier_ratio = ransac_res.inlier_ratio

        if ransac_res.success:
            inlier_pts = np.float32(
                [ref_feat.keypoints[mm.queryIdx].pt for mm, keep in
                 zip(match_res.good_matches, ransac_res.inlier_mask) if keep]
            )
            spatial_res = analyze_spatial_distribution(inlier_pts, reference.shape, grid_size=grid_size)
            m.spatial_coverage_pct = spatial_res.spatial_coverage_pct
            m.distribution_score = spatial_res.distribution_score
            m.success = True
        else:
            m.success = False
            m.failure_reason = ransac_res.message

    m.runtime_sec = t.elapsed
    return m


def run_scale_experiment(reference: np.ndarray, source: np.ndarray,
                          scale_factors=SCALE_FACTORS, **kwargs) -> list[PipelineMetrics]:
    results = []
    h, w = source.shape[:2]
    for s in scale_factors:
        new_w, new_h = max(32, int(w * s)), max(32, int(h * s))
        scaled = cv2.resize(source, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # resize back to original canvas so it still overlaps the reference frame
        scaled_canvas = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        label = f"Scale {s:.2f}x"
        results.append(_run_baseline_pipeline(reference, scaled_canvas, label, **kwargs))
    return results


def run_illumination_experiment(reference: np.ndarray, source: np.ndarray,
                                 presets=None, **kwargs) -> list[PipelineMetrics]:
    presets = presets or ILLUMINATION_PRESETS
    results = []
    for label, params in presets.items():
        varied = apply_illumination(source, brightness_delta=params["brightness_delta"], gamma=params["gamma"])
        results.append(_run_baseline_pipeline(reference, varied, f"Illumination: {label}", **kwargs))
    return results
