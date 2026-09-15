"""Feature matching: BFMatcher + KNN + Lowe's ratio test."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class MatchResult:
    candidate_matches: list       # all KNN pairs found (list of 2-tuples of cv2.DMatch)
    good_matches: list            # matches surviving Lowe ratio test
    num_candidates: int
    num_good: int


class InsufficientFeaturesError(Exception):
    pass


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75, k: int = 2) -> MatchResult:
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        raise InsufficientFeaturesError(
            "Not enough descriptors to match (need at least 2 keypoints in each image)."
        )

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=k)

    # guard against pairs where fewer than k neighbours were returned
    candidate_matches = [pair for pair in knn_matches if len(pair) == 2]

    good_matches = []
    for m, n in candidate_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return MatchResult(
        candidate_matches=candidate_matches,
        good_matches=good_matches,
        num_candidates=len(candidate_matches),
        num_good=len(good_matches),
    )


def draw_matches(img1, kp1, img2, kp2, matches, max_display: int = 100):
    matches_sorted = sorted(matches, key=lambda m: m.distance)[:max_display]
    return cv2.drawMatches(
        img1, kp1, img2, kp2, matches_sorted, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
