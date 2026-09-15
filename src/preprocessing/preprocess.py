"""
Image preprocessing stage of the MoonMatrix pipeline.

All steps operate on single-channel uint8 arrays and are individually
toggleable, matching the "let the user enable/disable" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class PreprocessConfig:
    normalize: bool = True
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    contrast_alpha: float = 1.0     # 1.0 = no change; simple linear contrast
    contrast_beta: int = 0          # brightness offset
    denoise: bool = False
    denoise_strength: int = 5
    resize_to: tuple | None = None  # (w, h) or None


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to full 0-255 range."""
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return img.astype(np.uint8)
    out = (img - mn) / (mx - mn) * 255.0
    return out.astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid=8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(img)


def adjust_contrast(img: np.ndarray, alpha=1.0, beta=0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def denoise_image(img: np.ndarray, strength=5) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def resize_image(img: np.ndarray, target_wh) -> np.ndarray:
    w, h = target_wh
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def run_preprocessing(img: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, list[str]]:
    """Apply the enabled steps in a fixed, sensible order. Returns (processed, log)."""
    log = []
    out = img.copy()

    if cfg.resize_to is not None:
        out = resize_image(out, cfg.resize_to)
        log.append(f"Resized to {cfg.resize_to}")

    if cfg.denoise:
        out = denoise_image(out, cfg.denoise_strength)
        log.append(f"Denoised (h={cfg.denoise_strength})")

    if cfg.normalize:
        out = normalize_image(out)
        log.append("Min-max normalized")

    if cfg.clahe:
        out = apply_clahe(out, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
        log.append(f"CLAHE (clip={cfg.clahe_clip_limit}, tile={cfg.clahe_tile_grid})")

    if cfg.contrast_alpha != 1.0 or cfg.contrast_beta != 0:
        out = adjust_contrast(out, cfg.contrast_alpha, cfg.contrast_beta)
        log.append(f"Contrast adjust (alpha={cfg.contrast_alpha}, beta={cfg.contrast_beta})")

    return out, log
