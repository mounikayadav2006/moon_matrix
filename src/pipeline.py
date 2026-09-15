"""
Pipeline orchestrator.

Runs the full MoonMatrix pipeline in one call, with automatic fallback
retries at each stage so a reasonable image pair reliably produces a
result instead of requiring the user to manually tune parameters.

This is still honest computation, not fabrication: every fallback is a
real re-run of the same algorithm with different (still principled)
parameters, and the final result records exactly which settings were
used and whether each stage genuinely succeeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.preprocessing.preprocess import PreprocessConfig, run_preprocessing
from src.features.sift import detect_sift_features, SiftResult
from src.matching.matcher import match_features, InsufficientFeaturesError, MatchResult
from src.geometry.ransac import run_ransac, RansacResult
from src.spatial.grid import analyze_spatial_distribution, SpatialAnalysisResult
from src.registration.register import register_source_to_reference, transform_error_against_ground_truth
from src.subpixel.refine import refine_correspondences
from src.evaluation.metrics import PipelineMetrics, Timer


@dataclass
class PipelineResult:
    reference_proc: np.ndarray
    source_proc: np.ndarray
    ref_feat: SiftResult
    src_feat: SiftResult
    match_res: MatchResult | None
    ransac_res: RansacResult | None
    spatial_res: SpatialAnalysisResult | None
    reg_res: object | None
    subpixel_res: object | None
    metrics: PipelineMetrics
    settings_used: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def run_full_pipeline(reference: np.ndarray, source: np.ndarray,
                       ground_truth_H: np.ndarray | None = None,
                       grid_size: int = 4) -> PipelineResult:
    """
    Run preprocessing -> SIFT -> matching -> RANSAC -> spatial analysis ->
    registration -> sub-pixel refinement, automatically retrying with more
    permissive (but still principled) parameters if a stage under-delivers.
    """
    notes = []
    settings_used = {}
    m = PipelineMetrics(experiment_name="Full Run")

    with Timer() as total_timer:
        # ---- 1. Preprocessing (fixed, sensible defaults) ----
        cfg = PreprocessConfig(normalize=True, clahe=True, clahe_clip_limit=2.0, denoise=False)
        ref_proc, _ = run_preprocessing(reference, cfg)
        src_proc, _ = run_preprocessing(source, cfg)

        # ---- 2. SIFT, with contrast-threshold fallback if too few keypoints ----
        contrast_thresholds = [0.04, 0.02, 0.01]
        ref_feat, src_feat = None, None
        for ct in contrast_thresholds:
            ref_feat = detect_sift_features(ref_proc, contrast_threshold=ct)
            src_feat = detect_sift_features(src_proc, contrast_threshold=ct)
            settings_used["sift_contrast_threshold"] = ct
            if ref_feat.num_keypoints >= 8 and src_feat.num_keypoints >= 8:
                break
            notes.append(f"Few keypoints at contrast threshold {ct}; relaxing detector sensitivity.")

        m.num_keypoints_ref = ref_feat.num_keypoints
        m.num_keypoints_src = src_feat.num_keypoints

        if ref_feat.num_keypoints < 4 or src_feat.num_keypoints < 4:
            notes.append("Too few keypoints detected even after relaxing thresholds. "
                         "Try higher-contrast or higher-resolution images.")
            m.success = False
            m.failure_reason = "Insufficient keypoints."
            m.runtime_sec = total_timer.elapsed if hasattr(total_timer, "elapsed") else 0.0
            return PipelineResult(ref_proc, src_proc, ref_feat, src_feat, None, None, None, None, None,
                                   m, settings_used, notes)

        # ---- 3. Matching, with ratio-threshold fallback ----
        ratios = [0.75, 0.8, 0.88]
        match_res = None
        for ratio in ratios:
            try:
                match_res = match_features(ref_feat.descriptors, src_feat.descriptors, ratio=ratio)
            except InsufficientFeaturesError as e:
                notes.append(str(e))
                break
            settings_used["lowe_ratio"] = ratio
            if match_res.num_good >= 8:
                break
            notes.append(f"Few good matches at ratio {ratio}; relaxing Lowe's ratio test.")

        if match_res is None or match_res.num_good < 4:
            m.num_candidate_matches = match_res.num_candidates if match_res else 0
            m.num_good_matches = match_res.num_good if match_res else 0
            notes.append("Too few reliable matches between the two images. "
                         "They may not overlap, or may need different preprocessing.")
            m.success = False
            m.failure_reason = "Insufficient good matches."
            m.runtime_sec = total_timer.elapsed if hasattr(total_timer, "elapsed") else 0.0
            return PipelineResult(ref_proc, src_proc, ref_feat, src_feat, match_res, None, None, None, None,
                                   m, settings_used, notes)

        m.num_candidate_matches = match_res.num_candidates
        m.num_good_matches = match_res.num_good

        # ---- 4. RANSAC, homography first, affine fallback ----
        ransac_res = run_ransac(ref_feat.keypoints, src_feat.keypoints, match_res.good_matches,
                                 transform_type="homography")
        settings_used["transform_type"] = "homography"
        if not ransac_res.success:
            notes.append("Homography verification failed; retrying with a simpler affine model.")
            ransac_res_affine = run_ransac(ref_feat.keypoints, src_feat.keypoints, match_res.good_matches,
                                            transform_type="affine")
            if ransac_res_affine.success or ransac_res_affine.num_inliers > ransac_res.num_inliers:
                ransac_res = ransac_res_affine
                settings_used["transform_type"] = "affine"

        m.num_inliers = ransac_res.num_inliers
        m.num_outliers = ransac_res.num_outliers
        m.inlier_ratio = ransac_res.inlier_ratio

        if not ransac_res.success:
            notes.append(ransac_res.message)
            m.success = False
            m.failure_reason = ransac_res.message
            m.runtime_sec = total_timer.elapsed if hasattr(total_timer, "elapsed") else 0.0
            return PipelineResult(ref_proc, src_proc, ref_feat, src_feat, match_res, ransac_res, None, None, None,
                                   m, settings_used, notes)

        # ---- 5. Spatial analysis ----
        inlier_pts = np.float32(
            [ref_feat.keypoints[mm.queryIdx].pt for mm, keep in zip(match_res.good_matches, ransac_res.inlier_mask) if keep]
        )
        spatial_res = analyze_spatial_distribution(inlier_pts, ref_proc.shape, grid_size=grid_size)
        m.spatial_coverage_pct = spatial_res.spatial_coverage_pct
        m.distribution_score = spatial_res.distribution_score

        # ---- 6. Registration ----
        reg_res = register_source_to_reference(ref_proc, src_proc, ransac_res.matrix, ransac_res.transform_type)
        if ground_truth_H is not None:
            m.registration_error_px = transform_error_against_ground_truth(
                ransac_res.matrix, ground_truth_H, ref_proc.shape
            )

        # ---- 7. Sub-pixel refinement ----
        src_pts = np.float32(
            [src_feat.keypoints[mm.trainIdx].pt for mm, keep in zip(match_res.good_matches, ransac_res.inlier_mask) if keep]
        )
        subpixel_res = refine_correspondences(ref_proc, src_proc, inlier_pts, src_pts, patch_size=32)
        if subpixel_res.num_refined > 0:
            m.refinement_error_px = subpixel_res.mean_displacement_px

        m.success = True

    m.runtime_sec = total_timer.elapsed
    return PipelineResult(ref_proc, src_proc, ref_feat, src_feat, match_res, ransac_res, spatial_res,
                           reg_res, subpixel_res, m, settings_used, notes)
