"""Pixel-array preprocessing transforms.

Pure functions that convert a decoded DICOM pixel array into a model-ready
tensor: channel conversion (grayscale <-> RGB), normalization to [0, 1], and
resizing to the model input resolution.

Requirements: 3.2, 3.6
"""

from __future__ import annotations

import numpy as np

GRAYSCALE = "grayscale"
RGB = "rgb"


def _to_2d(img: np.ndarray) -> np.ndarray:
    """Collapse a trailing singleton channel so we always start from 2D."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def to_channels(img: np.ndarray, target: str) -> np.ndarray:
    """Convert an image to the target channel representation.

    ``target="grayscale"`` yields a ``(H, W, 1)`` array; ``target="rgb"``
    yields ``(H, W, 3)`` by replicating the single channel. Spatial
    dimensions are preserved, so a grayscale -> rgb -> grayscale round trip
    keeps the original height and width.

    Requirements: 3.2
    """
    arr = np.asarray(img)

    if target == GRAYSCALE:
        if arr.ndim == 2:
            return arr[..., np.newaxis]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr
        if arr.ndim == 3 and arr.shape[-1] == 3:
            # Luminosity-weighted collapse to a single channel.
            gray = arr[..., :3].mean(axis=-1)
            return gray[..., np.newaxis].astype(arr.dtype)
        raise ValueError(f"Unsupported image shape for grayscale: {arr.shape}")

    if target == RGB:
        base = _to_2d(arr)
        if base.ndim != 2:
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return arr
            raise ValueError(f"Unsupported image shape for rgb: {arr.shape}")
        return np.repeat(base[..., np.newaxis], 3, axis=-1)

    raise ValueError(f"Unknown target channel representation: {target!r}")


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale pixel values into [0, 1] via min-max normalization.

    Relative ordering of intensities is preserved. A constant image maps to
    all zeros (no division by zero).

    Requirements: 3.6
    """
    arr = np.asarray(img, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    span = hi - lo
    if span == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / span


def resize(img: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize a 2D or channel-last image to ``size`` = (height, width).

    Uses Pillow for high-quality resampling and preserves the channel count.

    Requirements: 3.2
    """
    from PIL import Image

    arr = np.asarray(img)
    target_h, target_w = size

    def _resize_plane(plane: np.ndarray) -> np.ndarray:
        # Pillow needs an 8-bit or float image; scale to uint8 for resampling
        # then return float to avoid clipping surprises downstream.
        p = plane.astype(np.float32)
        lo, hi = float(p.min()), float(p.max())
        if hi > lo:
            p8 = ((p - lo) / (hi - lo) * 255.0).astype(np.uint8)
        else:
            p8 = np.zeros_like(p, dtype=np.uint8)
        im = Image.fromarray(p8).resize((target_w, target_h), Image.BILINEAR)
        out = np.asarray(im, dtype=np.float32)
        # Restore original dynamic range.
        if hi > lo:
            out = out / 255.0 * (hi - lo) + lo
        else:
            out = np.full_like(out, lo)
        return out

    if arr.ndim == 2:
        return _resize_plane(arr)
    if arr.ndim == 3:
        channels = [_resize_plane(arr[..., c]) for c in range(arr.shape[-1])]
        return np.stack(channels, axis=-1)
    raise ValueError(f"Unsupported image shape for resize: {arr.shape}")
