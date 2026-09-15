"""RANSAC geometric verification of feature matches."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class RansacResult:
    success: bool
    transform_type: str                  # "homography" or "affine"
    matrix: np.ndarray | None
    inlier_mask: np.ndarray | None       # boolean array, one per input match
    num_inliers: int
    num_outliers: int
    inlier_ratio: float
    message: str


MIN_MATCHES_HOMOGRAPHY = 4
MIN_MATCHES_AFFINE = 3


def run_ransac(
    kp1, kp2, good_matches, transform_type: str = "homography",
    ransac_reproj_threshold: float = 5.0, min_inliers: int = 8,
) -> RansacResult:
    """
    Estimate a geometric transform from reference keypoints (kp1) to source
    keypoints (kp2) using the given good matches, verified with RANSAC.

    Rejects (success=False) rather than forcing a result when there are too
    few matches or too few inliers.
    """
    n = len(good_matches)
    min_required = MIN_MATCHES_HOMOGRAPHY if transform_type == "homography" else MIN_MATCHES_AFFINE

    if n < min_required:
        return RansacResult(
            success=False, transform_type=transform_type, matrix=None, inlier_mask=None,
            num_inliers=0, num_outliers=n, inlier_ratio=0.0,
            message=f"Only {n} good matches found; need at least {min_required} for {transform_type}.",
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if transform_type == "homography":
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    elif transform_type == "affine":
        matrix, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_threshold
        )
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    if matrix is None or mask is None:
        return RansacResult(
            success=False, transform_type=transform_type, matrix=None, inlier_mask=None,
            num_inliers=0, num_outliers=n, inlier_ratio=0.0,
            message="RANSAC failed to find a consistent geometric transform.",
        )

    mask = mask.ravel().astype(bool)
    num_inliers = int(mask.sum())
    num_outliers = n - num_inliers
    inlier_ratio = num_inliers / n if n > 0 else 0.0

    if num_inliers < min_inliers:
        return RansacResult(
            success=False, transform_type=transform_type, matrix=matrix, inlier_mask=mask,
            num_inliers=num_inliers, num_outliers=num_outliers, inlier_ratio=inlier_ratio,
            message=f"Only {num_inliers} inliers found; below minimum threshold of {min_inliers}. "
                    f"Registration rejected rather than forced.",
        )

    return RansacResult(
        success=True, transform_type=transform_type, matrix=matrix, inlier_mask=mask,
        num_inliers=num_inliers, num_outliers=num_outliers, inlier_ratio=inlier_ratio,
        message="RANSAC verification succeeded.",
    )


def draw_inliers_outliers(img1, kp1, img2, kp2, good_matches, inlier_mask, max_display=150):
    """Draw inlier matches in green and outlier matches in red."""
    import cv2 as _cv2
    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask) if keep]
    outlier_matches = [m for m, keep in zip(good_matches, inlier_mask) if not keep]

    inlier_matches = sorted(inlier_matches, key=lambda m: m.distance)[:max_display]
    outlier_matches = sorted(outlier_matches, key=lambda m: m.distance)[:max_display]

    canvas = _cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        matchColor=(0, 200, 0), singlePointColor=None,
        flags=_cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    canvas2 = _cv2.drawMatches(
        img1, kp1, img2, kp2, outlier_matches, None,
        matchColor=(0, 0, 220), singlePointColor=None,
        flags=_cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    # blend: overlay red outlier lines onto the green inlier canvas
    blended = _cv2.addWeighted(canvas, 1.0, canvas2, 0.5, 0)
    return blended
