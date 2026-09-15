"""
Sub-pixel correspondence refinement — PROTOTYPE.

This module refines a small number of initial (integer-pixel) correspondence
points using local phase correlation on small patches around each point. This
is a genuine, working sub-pixel technique (OpenCV's cv2.phaseCorrelate), but
it is a *local, patch-based* estimate — it is NOT a validated, production
sub-pixel geodetic registration pipeline. Labelled honestly as a prototype
per the project requirements.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class SubpixelRefinementResult:
    initial_points: np.ndarray      # Nx2 (reference-image points)
    refined_points: np.ndarray      # Nx2 (reference-image points, sub-pixel)
    displacements: np.ndarray       # Nx2, refined - initial
    mean_displacement_px: float
    patch_size: int
    num_refined: int
    skipped: int


def refine_correspondences(reference: np.ndarray, source: np.ndarray,
                            ref_points: np.ndarray, src_points: np.ndarray,
                            patch_size: int = 32) -> SubpixelRefinementResult:
    """
    For each (ref_point, src_point) pair, extract a patch around src_point in
    `source` and around ref_point in `reference`, and use phase correlation to
    estimate the sub-pixel shift that best aligns them. The refined point is
    ref_point + shift.
    """
    half = patch_size // 2
    h, w = reference.shape[:2]
    sh, sw = source.shape[:2]

    initial, refined, skipped = [], [], 0

    ref_f = reference.astype(np.float32)
    src_f = source.astype(np.float32)

    hann = cv2.createHanningWindow((patch_size, patch_size), cv2.CV_32F)

    for (rx, ry), (sx, sy) in zip(ref_points, src_points):
        rx, ry, sx, sy = float(rx), float(ry), float(sx), float(sy)
        ri, rj = int(round(ry)), int(round(rx))
        si, sj = int(round(sy)), int(round(sx))

        if (ri - half < 0 or ri + half >= h or rj - half < 0 or rj + half >= w or
                si - half < 0 or si + half >= sh or sj - half < 0 or sj + half >= sw):
            skipped += 1
            continue

        patch_ref = ref_f[ri - half:ri + half, rj - half:rj + half]
        patch_src = src_f[si - half:si + half, sj - half:sj + half]

        if patch_ref.shape != (patch_size, patch_size) or patch_src.shape != (patch_size, patch_size):
            skipped += 1
            continue

        try:
            (dx, dy), _response = cv2.phaseCorrelate(patch_ref, patch_src, hann)
        except cv2.error:
            skipped += 1
            continue

        initial.append([rx, ry])
        # dx, dy is the shift needed to align patch_src to patch_ref; apply it
        # as a sub-pixel correction to the reference-frame point
        refined.append([rx - dx, ry - dy])

    initial = np.array(initial, dtype=np.float64) if initial else np.zeros((0, 2))
    refined = np.array(refined, dtype=np.float64) if refined else np.zeros((0, 2))
    displacements = refined - initial if len(initial) else np.zeros((0, 2))
    mean_disp = float(np.mean(np.linalg.norm(displacements, axis=1))) if len(displacements) else 0.0

    return SubpixelRefinementResult(
        initial_points=initial, refined_points=refined, displacements=displacements,
        mean_displacement_px=mean_disp, patch_size=patch_size,
        num_refined=len(initial), skipped=skipped,
    )
