"""Small helpers for loading/validating images consistently across the app."""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


class InvalidImageError(Exception):
    pass


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Load a Streamlit UploadedFile into a grayscale uint8 numpy array."""
    try:
        pil_img = Image.open(uploaded_file).convert("L")
        arr = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        raise InvalidImageError(f"Could not read uploaded image: {e}")

    validate_image(arr)
    return arr


def load_image_from_path(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise InvalidImageError(f"Could not read image at path: {path}")
    validate_image(arr)
    return arr


def validate_image(arr: np.ndarray) -> None:
    if arr is None or arr.size == 0:
        raise InvalidImageError("Image is empty.")
    if arr.ndim != 2:
        raise InvalidImageError("Image must be single-channel grayscale after conversion.")
    h, w = arr.shape
    if h < 32 or w < 32:
        raise InvalidImageError(f"Image too small ({w}x{h}); minimum 32x32 required.")


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)


def resize_to_match(img: np.ndarray, target_shape) -> np.ndarray:
    h, w = target_shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
