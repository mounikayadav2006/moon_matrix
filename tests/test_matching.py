import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.utils.sample_data import make_reference_and_source
from src.features.sift import detect_sift_features
from src.matching.matcher import match_features, InsufficientFeaturesError


def test_sift_detects_keypoints_on_synthetic_image():
    ref, src, _ = make_reference_and_source()
    feat = detect_sift_features(ref)
    assert feat.num_keypoints > 0
    assert feat.descriptors is not None
    assert feat.descriptors.shape[0] == feat.num_keypoints


def test_matching_produces_good_matches_for_related_images():
    ref, src, _ = make_reference_and_source()
    rf = detect_sift_features(ref)
    sf = detect_sift_features(src)
    result = match_features(rf.descriptors, sf.descriptors)
    assert result.num_candidates > 0
    assert result.num_good > 0
    assert result.num_good <= result.num_candidates


def test_matching_raises_on_insufficient_descriptors():
    empty = np.zeros((1, 128), dtype=np.float32)
    with pytest.raises(InsufficientFeaturesError):
        match_features(empty, empty)


def test_matching_raises_on_none_descriptors():
    with pytest.raises(InsufficientFeaturesError):
        match_features(None, None)
