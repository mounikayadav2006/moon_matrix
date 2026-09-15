"""Image registration: warp the source image into the reference frame."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class RegistrationResult:
    registered_image: np.ndarray
    overlay: np.ndarray
    difference: np.ndarray
    transform_type: str


def register_source_to_reference(reference: np.ndarray, source: np.ndarray,
                                  matrix: np.ndarray, transform_type: str) -> RegistrationResult:
    h, w = reference.shape[:2]

    if transform_type == "homography":
        registered = cv2.warpPerspective(source, matrix, (w, h), flags=cv2.INTER_LINEAR)
    elif transform_type == "affine":
        registered = cv2.warpAffine(source, matrix, (w, h), flags=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    overlay = make_overlay(reference, registered)
    difference = make_difference(reference, registered)

    return RegistrationResult(
        registered_image=registered, overlay=overlay, difference=difference,
        transform_type=transform_type,
    )


def make_overlay(reference: np.ndarray, registered: np.ndarray) -> np.ndarray:
    """Reference in green channel, registered source in red channel -> misalignment pops out as color fringing."""
    h, w = reference.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = reference       # green = reference
    overlay[:, :, 2] = registered      # red = registered source
    overlay[:, :, 0] = 0
    return overlay


def make_difference(reference: np.ndarray, registered: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(reference, registered)
    return diff


def transform_error_against_ground_truth(estimated_H: np.ndarray, ground_truth_H: np.ndarray,
                                          image_shape) -> float:
    """
    Mean corner-displacement error (in pixels) between applying the estimated
    transform and the known ground-truth transform to the four image corners.
    Only meaningful for synthetic/demo data where ground truth exists.
    """
    h, w = image_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    est_pts = cv2.perspectiveTransform(corners, _to_3x3(estimated_H))
    gt_pts = cv2.perspectiveTransform(corners, _to_3x3(ground_truth_H))

    errs = np.linalg.norm(est_pts - gt_pts, axis=2).flatten()
    return float(np.mean(errs))


def _to_3x3(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (2, 3):
        return np.vstack([matrix, [0, 0, 1]])
    raise ValueError(f"Unexpected matrix shape: {matrix.shape}")
