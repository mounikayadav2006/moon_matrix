import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.spatial.grid import analyze_spatial_distribution


def test_uniform_points_give_full_coverage_and_high_score():
    # one point roughly centered in each of a 4x4 grid over a 400x400 image
    pts = []
    for r in range(4):
        for c in range(4):
            pts.append([c * 100 + 50, r * 100 + 50])
    pts = np.array(pts, dtype=np.float32)

    result = analyze_spatial_distribution(pts, (400, 400), grid_size=4)
    assert result.occupied_cells == 16
    assert result.spatial_coverage_pct == 100.0
    assert result.distribution_score > 0.9


def test_clustered_points_give_low_coverage():
    pts = np.array([[10, 10], [12, 12], [15, 15], [11, 14]], dtype=np.float32)
    result = analyze_spatial_distribution(pts, (400, 400), grid_size=4)
    assert result.occupied_cells == 1
    assert result.spatial_coverage_pct == 100.0 / 16
    assert result.distribution_score == 0.0  # all mass in one cell -> zero entropy


def test_empty_points_give_zero_coverage():
    pts = np.zeros((0, 2), dtype=np.float32)
    result = analyze_spatial_distribution(pts, (400, 400), grid_size=4)
    assert result.occupied_cells == 0
    assert result.spatial_coverage_pct == 0.0
    assert result.distribution_score == 0.0
