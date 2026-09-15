import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.preprocessing.preprocess import (
    PreprocessConfig, run_preprocessing, normalize_image, apply_clahe, adjust_contrast,
)


def _make_test_image():
    rng = np.random.default_rng(0)
    return (rng.random((128, 128)) * 100 + 50).astype(np.uint8)


def test_normalize_stretches_to_full_range():
    img = _make_test_image()
    out = normalize_image(img)
    assert out.min() == 0
    assert out.max() == 255


def test_clahe_runs_without_error():
    img = _make_test_image()
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_contrast_adjust_changes_values():
    img = _make_test_image()
    out = adjust_contrast(img, alpha=1.5, beta=10)
    assert not np.array_equal(img, out)


def test_run_preprocessing_pipeline_all_enabled():
    img = _make_test_image()
    cfg = PreprocessConfig(normalize=True, clahe=True, contrast_alpha=1.1, contrast_beta=5, denoise=False)
    out, log = run_preprocessing(img, cfg)
    assert out.shape == img.shape
    assert len(log) >= 2


def test_run_preprocessing_all_disabled_is_noop():
    img = _make_test_image()
    cfg = PreprocessConfig(normalize=False, clahe=False, contrast_alpha=1.0, contrast_beta=0, denoise=False)
    out, log = run_preprocessing(img, cfg)
    assert np.array_equal(out, img)
    assert log == []
