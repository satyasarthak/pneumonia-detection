"""Exploratory data analysis helpers.

Sample and annotate images per class, compute the class distribution, and
flag class imbalance.

Requirements: 2.1, 2.2, 2.3, 2.4
"""

from __future__ import annotations

import io

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


# --- Extended EDA: DICOM metadata + pixel statistics ------------------------
# These functions sample images per class (decoding every one of the ~26k
# files is slow) and extract header metadata and pixel statistics used for the
# richer EDA plots in the report/notebook.

_META_FIELDS = ("PatientAge", "PatientSex", "ViewPosition", "Rows", "Columns")


def collect_metadata(
    df: pd.DataFrame,
    image_source,
    per_class: int = 400,
    seed: int = 42,
    label_col: str = "class_label",
    with_pixel_stats: bool = True,
) -> pd.DataFrame:
    """Sample images per class and extract DICOM metadata + pixel statistics.

    Returns a tidy DataFrame with one row per sampled image and columns:
    patientId, class_label, age, sex, view, rows, cols, mean_intensity,
    std_intensity. Decode/parse failures are skipped.

    Requirements: 2.4 (supports imbalance / distribution analysis)
    """
    import pydicom

    owns_source = isinstance(image_source, str)
    source = ImageSource(image_source) if owns_source else image_source
    rng = np.random.default_rng(seed)

    records: list[dict] = []
    try:
        for cls in CLASSES:
            pids = df.loc[df[label_col] == cls, "patientId"].tolist()
            rng.shuffle(pids)
            taken = 0
            for pid in pids:
                if taken >= per_class:
                    break
                try:
                    raw = source.read_bytes(str(pid))
                    ds = pydicom.dcmread(io.BytesIO(raw))
                except Exception:  # noqa: BLE001
                    continue

                def _get(field):
                    return getattr(ds, field, None)

                age = _get("PatientAge")
                try:
                    # Ages look like "35" or "035Y"; keep the leading integer.
                    age = int(str(age).rstrip("Y").lstrip("0") or 0) if age else None
                except (ValueError, TypeError):
                    age = None

                row = {
                    "patientId": pid,
                    "class_label": cls,
                    "age": age,
                    "sex": _get("PatientSex"),
                    "view": _get("ViewPosition"),
                    "rows": int(_get("Rows")) if _get("Rows") else None,
                    "cols": int(_get("Columns")) if _get("Columns") else None,
                }
                if with_pixel_stats:
                    try:
                        arr = ds.pixel_array.astype(np.float32)
                        row["mean_intensity"] = float(arr.mean())
                        row["std_intensity"] = float(arr.std())
                    except Exception:  # noqa: BLE001
                        row["mean_intensity"] = None
                        row["std_intensity"] = None
                records.append(row)
                taken += 1
    finally:
        if owns_source:
            source.close()

    return pd.DataFrame.from_records(records)


def plot_class_distribution_pie(dist: dict[str, int]):
    """Pie chart of class shares alongside counts (Req 2.3)."""
    import matplotlib.pyplot as plt

    labels = list(dist.keys())
    sizes = list(dist.values())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=["#4c72b0", "#dd8452", "#55a868"],
        textprops={"fontsize": 9},
    )
    ax.set_title("Class share of patients")
    fig.tight_layout()
    return fig


def plot_age_distribution(meta: pd.DataFrame):
    """Patient-age distribution overall and by class (box plot)."""
    import matplotlib.pyplot as plt

    valid = meta.dropna(subset=["age"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(valid["age"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Patient age distribution")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")

    groups = [valid.loc[valid["class_label"] == c, "age"].values for c in CLASSES]
    axes[1].boxplot(groups, tick_labels=[c.split(" /")[0] for c in CLASSES])
    axes[1].set_title("Age by class")
    axes[1].set_ylabel("Age (years)")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def plot_sex_and_view(meta: pd.DataFrame):
    """Grouped bar charts: patient sex and view position by class."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, field, title in (
        (axes[0], "sex", "Patient sex by class"),
        (axes[1], "view", "View position by class"),
    ):
        ct = (
            meta.dropna(subset=[field])
            .groupby(["class_label", field])
            .size()
            .unstack(fill_value=0)
        )
        ct = ct.reindex([c for c in CLASSES if c in ct.index])
        ct.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.set_xticklabels([c.split(" /")[0] for c in ct.index], rotation=15)
        ax.legend(title=field)
    fig.tight_layout()
    return fig


def plot_intensity_by_class(meta: pd.DataFrame):
    """Mean pixel-intensity distribution per class (overlaid histograms)."""
    import matplotlib.pyplot as plt

    valid = meta.dropna(subset=["mean_intensity"])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for cls, color in zip(CLASSES, colors):
        vals = valid.loc[valid["class_label"] == cls, "mean_intensity"].values
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.55, label=cls.split(" /")[0], color=color)
    ax.set_title("Mean pixel intensity by class")
    ax.set_xlabel("Mean pixel intensity")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_image_dimensions(meta: pd.DataFrame):
    """Distribution of raw image dimensions (are all images the same size?)."""
    import matplotlib.pyplot as plt

    valid = meta.dropna(subset=["rows", "cols"])
    fig, ax = plt.subplots(figsize=(6, 5))
    dims = valid.apply(lambda r: f"{int(r['rows'])}x{int(r['cols'])}", axis=1)
    counts = dims.value_counts()
    ax.bar(counts.index.astype(str), counts.values, color="slategray")
    ax.set_title("Raw image dimensions")
    ax.set_xlabel("Rows x Columns")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def mean_image_per_class(
    df: pd.DataFrame,
    image_source,
    per_class: int = 100,
    size: tuple[int, int] = (128, 128),
    seed: int = 42,
    label_col: str = "class_label",
) -> dict[str, np.ndarray]:
    """Compute the average (mean) X-ray per class.

    Averaging many images reveals where signal concentrates - the mean
    pneumonia image tends to show diffuse lower-lung haze versus the crisper
    mean Normal image.
    """
    from .preprocess import normalize, resize

    owns_source = isinstance(image_source, str)
    source = ImageSource(image_source) if owns_source else image_source
    rng = np.random.default_rng(seed)

    means: dict[str, np.ndarray] = {}
    try:
        for cls in CLASSES:
            pids = df.loc[df[label_col] == cls, "patientId"].tolist()
            rng.shuffle(pids)
            acc = np.zeros(size, dtype=np.float64)
            taken = 0
            for pid in pids:
                if taken >= per_class:
                    break
                try:
                    arr = load_dicom(source, str(pid))
                except DicomDecodeError:
                    continue
                acc += normalize(resize(arr, size))
                taken += 1
            means[cls] = (acc / taken) if taken else acc
    finally:
        if owns_source:
            source.close()
    return means


def plot_mean_images(means: dict[str, np.ndarray]):
    """Show the mean image per class side by side."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(means), figsize=(4 * len(means), 4))
    axes = np.atleast_1d(axes)
    for ax, (cls, img) in zip(axes, means.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Mean image\n{cls}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig
