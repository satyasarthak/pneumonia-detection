"""Memory-efficient batched image generator.

``XrayBatchGenerator`` subclasses ``keras.utils.Sequence`` and loads only the
images referenced by the current batch, so the ~53k-file dataset never has to
sit in memory at once. Each sample is decoded, resized, normalized, and
channel-converted on the fly.

Requirements: 3.7, 3.2, 3.6
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CLASS_TO_INDEX, NUM_CLASSES
from .images import DicomDecodeError, ImageSource, load_dicom
from .preprocess import GRAYSCALE, RGB, normalize, resize, to_channels


def _keras_sequence_base():
    """Return keras.utils.Sequence, or object if TF is unavailable.

    Importing TensorFlow is expensive and unnecessary for the pure data-layer
    tests, so the base class is resolved lazily. When TF is not installed the
    generator still works as a plain iterable for those tests.
    """
    try:
        from tensorflow.keras.utils import Sequence  # type: ignore

        return Sequence
    except Exception:  # noqa: BLE001
        return object


class XrayBatchGenerator(_keras_sequence_base()):  # type: ignore[misc]
    """Yield ``(batch_images, batch_labels)`` loading images incrementally.

    Parameters
    ----------
    records:
        DataFrame with ``patientId`` and ``class_label`` columns.
    image_source:
        Path (dir or zip) or an :class:`ImageSource`.
    batch_size:
        Number of samples per batch.
    target_channels:
        1 for grayscale (baseline CNN) or 3 for RGB (transfer models).
    image_size:
        (height, width) to resize each image to.
    augment:
        When True, apply light random horizontal flips (one lever for class
        imbalance). Off by default and never used for val/test.
    shuffle:
        Shuffle sample order each epoch (train only).
    seed:
        Seed for the shuffle RNG.

    Requirements: 3.7, 3.2, 3.6
    """

    def __init__(
        self,
        records: pd.DataFrame,
        image_source,
        batch_size: int = 32,
        target_channels: int = 3,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = False,
        shuffle: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        # Forward worker/queue kwargs to keras.utils.Sequence when available.
        try:
            super().__init__(**kwargs)
        except TypeError:
            pass
        self.records = records.reset_index(drop=True)
        self._owns_source = isinstance(image_source, str)
        self.source = ImageSource(image_source) if self._owns_source else image_source
        self.batch_size = int(batch_size)
        self.target_channels = int(target_channels)
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.records))
        if self.shuffle:
            self._rng.shuffle(self.indices)

    def __len__(self) -> int:
        # Ceil division so the final (possibly smaller) batch is included.
        return int(np.ceil(len(self.records) / self.batch_size))

    def _load_one(self, row) -> np.ndarray:
        arr = load_dicom(self.source, str(row["patientId"]))
        arr = resize(arr, self.image_size)
        arr = normalize(arr)
        target = GRAYSCALE if self.target_channels == 1 else RGB
        arr = to_channels(arr, target)
        if self.augment:
            arr = self._augment(arr)
        return arr.astype(np.float32)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Light, clinically-plausible augmentation for chest X-rays.

        Uses only transforms that preserve anatomical validity: horizontal
        flip (left/right is acceptable for this task), small rotation, small
        zoom, and mild brightness jitter. No vertical flip (upside-down chest
        X-rays do not occur).
        """
        # Horizontal flip.
        if self._rng.random() < 0.5:
            arr = arr[:, ::-1, :]

        # Small rotation (+/- ~10 degrees) and zoom via PIL affine.
        if self._rng.random() < 0.5:
            from PIL import Image

            angle = float(self._rng.uniform(-10, 10))
            zoom = float(self._rng.uniform(0.9, 1.1))
            h, w = arr.shape[:2]
            channels = []
            for c in range(arr.shape[-1]):
                plane = (arr[..., c] * 255.0).clip(0, 255).astype(np.uint8)
                im = Image.fromarray(plane)
                im = im.rotate(angle, resample=Image.BILINEAR)
                # Zoom by resizing then center-cropping/padding back to (w, h).
                zw, zh = max(1, int(w * zoom)), max(1, int(h * zoom))
                im = im.resize((zw, zh), Image.BILINEAR)
                canvas = Image.new("L", (w, h))
                canvas.paste(im, ((w - zw) // 2, (h - zh) // 2))
                channels.append(np.asarray(canvas, dtype=np.float32) / 255.0)
            arr = np.stack(channels, axis=-1)

        # Mild brightness jitter.
        if self._rng.random() < 0.5:
            factor = float(self._rng.uniform(0.9, 1.1))
            arr = np.clip(arr * factor, 0.0, 1.0)

        return arr

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.records))
        batch_indices = self.indices[start:end]

        images: list[np.ndarray] = []
        labels: list[int] = []
        for i in batch_indices:
            row = self.records.iloc[int(i)]
            try:
                images.append(self._load_one(row))
                labels.append(CLASS_TO_INDEX[row["class_label"]])
            except DicomDecodeError:
                # Corrupt file: skip like a missing image.
                continue

        x = np.stack(images, axis=0) if images else np.empty(
            (0, *self.image_size, self.target_channels), dtype=np.float32
        )
        y = _one_hot(labels)
        return x, y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self.indices)


def _one_hot(labels: list[int]) -> np.ndarray:
    y = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    for i, label in enumerate(labels):
        y[i, label] = 1.0
    return y
