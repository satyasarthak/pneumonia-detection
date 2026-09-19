"""Exploratory data analysis helpers.

Sample and annotate images per class, compute the class distribution, and
flag class imbalance.

Requirements: 2.1, 2.2, 2.3, 2.4
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CLASSES
from .images import DicomDecodeError, ImageSource, load_dicom


def class_distribution(df: pd.DataFrame, label_col: str = "class_label") -> dict[str, int]:
    """Count records per Class_Label.

    Every canonical class is present in the result (0 when absent) so the
    returned mapping is stable across datasets.

    Requirements: 2.3
    """
    counts = df[label_col].value_counts().to_dict()
    return {cls: int(counts.get(cls, 0)) for cls in CLASSES}


def imbalance_report(dist: dict[str, int], threshold: float = 1.5) -> dict:
    """Flag imbalance when the max/min class ratio exceeds ``threshold``.

    Returns a dict with the majority/minority classes, the imbalance ratio,
    a boolean flag, and a human-readable observation string.

    Requirements: 2.4
    """
    present = {k: v for k, v in dist.items() if v > 0}
    if not present:
        return {
            "imbalanced": False,
            "ratio": 0.0,
            "majority_class": None,
            "minority_class": None,
            "observation": "No records available to assess class balance.",
        }

    majority_class = max(present, key=present.get)
    minority_class = min(present, key=present.get)
    max_count = present[majority_class]
    min_count = present[minority_class]
    ratio = max_count / min_count if min_count > 0 else float("inf")
    imbalanced = ratio > threshold

    if imbalanced:
        observation = (
            f"Class imbalance detected: '{majority_class}' ({max_count}) has "
            f"{ratio:.2f}x the samples of '{minority_class}' ({min_count}). "
            "Consider class weighting or augmentation during training."
        )
    else:
        observation = (
            f"Classes are reasonably balanced (max/min ratio {ratio:.2f} "
            f"within threshold {threshold})."
        )

    return {
        "imbalanced": imbalanced,
        "ratio": ratio,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "counts": dist,
        "observation": observation,
    }


def sample_images_per_class(
    df: pd.DataFrame,
    image_source,
    n: int = 5,
    seed: int = 42,
    label_col: str = "class_label",
) -> dict[str, list[np.ndarray]]:
    """Randomly sample up to ``n`` decoded images per Class_Label.

    Returns a mapping ``class_label -> [pixel_array, ...]``. Corrupt files are
    skipped. Callers annotate each image with its class when plotting
    (see :func:`plot_samples`).

    Requirements: 2.1, 2.2
    """
    owns_source = isinstance(image_source, str)
    source = ImageSource(image_source) if owns_source else image_source
    rng = np.random.default_rng(seed)

    result: dict[str, list[np.ndarray]] = {}
    try:
        for cls in CLASSES:
            subset = df.loc[df[label_col] == cls, "patientId"].tolist()
            rng.shuffle(subset)
            images: list[np.ndarray] = []
            for pid in subset:
                if len(images) >= n:
                    break
                try:
                    images.append(load_dicom(source, str(pid)))
                except DicomDecodeError:
                    continue
            result[cls] = images
    finally:
        if owns_source:
            source.close()

    return result


def plot_samples(samples: dict[str, list[np.ndarray]], n: int = 5):
    """Render a grid of sample images, each annotated with its Class_Label.

    Returns the matplotlib Figure so it can be displayed or saved.

    Requirements: 2.1, 2.2
    """
    import matplotlib.pyplot as plt

    n_classes = len(samples)
    fig, axes = plt.subplots(n_classes, n, figsize=(3 * n, 3 * n_classes))
    axes = np.atleast_2d(axes)

    for r, (cls, images) in enumerate(samples.items()):
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(images):
                ax.imshow(images[c], cmap="gray")
                ax.set_title(cls, fontsize=9)
    fig.tight_layout()
    return fig


def plot_class_distribution(dist: dict[str, int]):
    """Bar chart of per-class counts.

    Requirements: 2.3
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(dist.keys()), list(dist.values()), color="steelblue")
    ax.set_ylabel("Number of patients")
    ax.set_title("Class distribution")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_before_after(raw: np.ndarray, processed: np.ndarray):
    """Show an image before and after preprocessing side by side.

    Requirements: 3.3
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("Before (raw DICOM)")
    axes[0].axis("off")

    disp = processed[..., 0] if processed.ndim == 3 else processed
    axes[1].imshow(disp, cmap="gray")
    axes[1].set_title("After (resized + normalized)")
    axes[1].axis("off")
    fig.tight_layout()
    return fig
