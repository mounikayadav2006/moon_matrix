import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.sample_data import make_reference_and_source
from src.features.sift import detect_sift_features
from src.matching.matcher import match_features
from src.geometry.ransac import run_ransac


def _full_pipeline():
    ref, src, H_gt = make_reference_and_source()
    rf = detect_sift_features(ref)
    sf = detect_sift_features(src)
    mr = match_features(rf.descriptors, sf.descriptors)
    return rf, sf, mr, H_gt


def test_ransac_succeeds_on_related_images_homography():
    rf, sf, mr, _ = _full_pipeline()
    result = run_ransac(rf.keypoints, sf.keypoints, mr.good_matches, transform_type="homography")
    assert result.success
    assert result.num_inliers > 0
    assert 0.0 <= result.inlier_ratio <= 1.0
    assert result.matrix.shape == (3, 3)


def test_ransac_succeeds_on_related_images_affine():
    rf, sf, mr, _ = _full_pipeline()
    result = run_ransac(rf.keypoints, sf.keypoints, mr.good_matches, transform_type="affine")
    assert result.success
    assert result.matrix.shape == (2, 3)


def test_ransac_rejects_when_too_few_matches():
    rf, sf, mr, _ = _full_pipeline()
    result = run_ransac(rf.keypoints, sf.keypoints, mr.good_matches[:2], transform_type="homography")
    assert not result.success
    assert result.num_inliers == 0


def test_ransac_rejects_on_unrelated_random_matches():
    import cv2
    import numpy as np
    rf, sf, mr, _ = _full_pipeline()
    # shuffle train indices to break the true correspondence -> should fail or have low inliers
    import random
    fake_matches = []
    for m in mr.good_matches[:15]:
        fake = cv2.DMatch(m.queryIdx, random.randint(0, len(sf.keypoints) - 1), m.distance)
        fake_matches.append(fake)
    result = run_ransac(rf.keypoints, sf.keypoints, fake_matches, transform_type="homography", min_inliers=8)
    # either RANSAC rejects it outright, or inlier ratio is low -- never silently "perfect"
    assert (not result.success) or result.inlier_ratio < 0.9
