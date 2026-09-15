"""SIFT feature detection wrapper."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class SiftResult:
    keypoints: list
    descriptors: np.ndarray | None
    num_keypoints: int


def detect_sift_features(img: np.ndarray, n_features: int = 0, contrast_threshold: float = 0.04,
                          edge_threshold: float = 10.0) -> SiftResult:
    """
    Run SIFT on a grayscale image.

    n_features=0 lets OpenCV keep all features it finds (no cap).
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )
    keypoints, descriptors = sift.detectAndCompute(img, None)
    n = len(keypoints) if keypoints is not None else 0
    return SiftResult(keypoints=keypoints, descriptors=descriptors, num_keypoints=n)


def draw_keypoints(img: np.ndarray, sift_result: SiftResult) -> np.ndarray:
    return cv2.drawKeypoints(
        img,
        sift_result.keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 255),
    )
