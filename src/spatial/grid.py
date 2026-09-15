"""Grid-based spatial distribution analysis of inlier correspondences."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class SpatialAnalysisResult:
    grid_size: int
    occupied_cells: int
    total_cells: int
    spatial_coverage_pct: float
    matches_per_cell: np.ndarray   # grid_size x grid_size
    distribution_score: float      # 0-1, higher = more evenly spread


def analyze_spatial_distribution(points_xy: np.ndarray, image_shape, grid_size: int = 4) -> SpatialAnalysisResult:
    """
    points_xy: Nx2 array of (x, y) pixel coordinates (e.g. inlier keypoints in
    the reference image).
    """
    h, w = image_shape[:2]
    total_cells = grid_size * grid_size
    counts = np.zeros((grid_size, grid_size), dtype=int)

    if points_xy is not None and len(points_xy) > 0:
        cell_w = w / grid_size
        cell_h = h / grid_size
        for x, y in points_xy:
            col = min(int(x // cell_w), grid_size - 1)
            row = min(int(y // cell_h), grid_size - 1)
            counts[row, col] += 1

    occupied = int((counts > 0).sum())
    coverage_pct = 100.0 * occupied / total_cells

    # distribution score: normalized entropy of the per-cell counts (0 = all in
    # one cell, 1 = perfectly uniform across occupied capacity)
    total_pts = counts.sum()
    if total_pts > 0:
        probs = counts.flatten() / total_pts
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
        max_entropy = np.log(total_cells)
        distribution_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        distribution_score = 0.0

    return SpatialAnalysisResult(
        grid_size=grid_size,
        occupied_cells=occupied,
        total_cells=total_cells,
        spatial_coverage_pct=coverage_pct,
        matches_per_cell=counts,
        distribution_score=distribution_score,
    )


def draw_grid_overlay(img: np.ndarray, points_xy: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Draw the grid lines and the correspondence points on a copy of the image."""
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = canvas.shape[:2]
    cell_w, cell_h = w / grid_size, h / grid_size

    for i in range(1, grid_size):
        x = int(i * cell_w)
        y = int(i * cell_h)
        cv2.line(canvas, (x, 0), (x, h), (80, 80, 80), 1)
        cv2.line(canvas, (0, y), (w, y), (80, 80, 80), 1)

    if points_xy is not None:
        for x, y in points_xy:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 255), -1)

    return canvas
