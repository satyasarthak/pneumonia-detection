"""Interim deliverable driver: Data Overview, EDA, and Preprocessing.

Runs tasks 1-6 and 8 end to end against the real RSNA data and writes the
figures required by the interim rubric into ``outputs/``:

  1. Data Overview       - import data, report shapes
  2. EDA                 - random images per class (annotated), class
                           distribution, imbalance observation
  3. Preprocessing demo  - DICOM decode -> resize -> normalize -> grayscale,
                           with a before/after figure
  4. Split summary       - stratified train/val/test with per-class counts

Usage
-----
    python run_eda.py

Assumes the dataset files sit next to this script:
    stage_2_detailed_class_info.csv
    stage_2_train_images.zip
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from src.eda import (
    class_distribution,
    imbalance_report,
    plot_before_after,
    plot_class_distribution,
    plot_samples,
    sample_images_per_class,
)
from src.images import ImageSource, load_dicom, resolve_images
from src.labels import deduplicate_patients, load_labels, restrict_labels
from src.preprocess import GRAYSCALE, normalize, resize, to_channels
from src.split import stratified_split

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_INFO_CSV = os.path.join(HERE, "stage_2_detailed_class_info.csv")
TRAIN_ZIP = os.path.join(HERE, "stage_2_train_images.zip")
OUT_DIR = os.path.join(HERE, "outputs")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. Data Overview (Req 1.1, 1.2, 1.3, 1.4) ---
    section("1. DATA OVERVIEW")
    labels = load_labels(CLASS_INFO_CSV)
    print(f"Raw label rows: {labels.shape}")

    deduped = deduplicate_patients(labels)
    print(f"After de-duplication (one label per patient): {deduped.shape}")

    clean = restrict_labels(deduped)
    print(f"After label restriction to 3 classes: {clean.shape}")

    # --- Resolve which patients have an image in the archive (Req 1.5) ---
    section("IMAGE RESOLUTION")
    present, missing = resolve_images(clean, TRAIN_ZIP)
    print(f"Records with an available image: {len(present)}")
    print(f"Records excluded (missing image): {len(missing)}")

    # --- 2. EDA (Req 2.1-2.4) ---
    section("2. EXPLORATORY DATA ANALYSIS")
    dist = class_distribution(present)
    print("Class distribution (patients per class):")
    for cls, count in dist.items():
        print(f"  {cls:<32} {count}")

    report = imbalance_report(dist)
    print("\nImbalance observation:")
    print(f"  {report['observation']}")

    fig = plot_class_distribution(dist)
    fig.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=120)
    print(f"  -> saved outputs/class_distribution.png")

    print("\nSampling random images per class (this decodes DICOMs)...")
    samples = sample_images_per_class(present, TRAIN_ZIP, n=5)
    fig = plot_samples(samples, n=5)
    fig.savefig(os.path.join(OUT_DIR, "sample_images_per_class.png"), dpi=120)
    print(f"  -> saved outputs/sample_images_per_class.png")

    # --- 3. Preprocessing demo with before/after (Req 3.1, 3.2, 3.3, 3.6) ---
    section("3. DATA PREPROCESSING")
    with ImageSource(TRAIN_ZIP) as src:
        sample_pid = str(present.iloc[0]["patientId"])
        raw = load_dicom(src, sample_pid)
    print(f"Sample patient: {sample_pid}")
    print(f"  Raw DICOM shape:   {raw.shape}, dtype: {raw.dtype}, "
          f"range: [{raw.min()}, {raw.max()}]")

    resized = resize(raw, (224, 224))
    normalized = normalize(resized)
    gray = to_channels(normalized, GRAYSCALE)
    print(f"  Processed shape:   {gray.shape}, dtype: {gray.dtype}, "
          f"range: [{gray.min():.3f}, {gray.max():.3f}]")

    fig = plot_before_after(raw, gray)
    fig.savefig(os.path.join(OUT_DIR, "preprocess_before_after.png"), dpi=120)
    print(f"  -> saved outputs/preprocess_before_after.png")

    # --- 4. Stratified split (Req 3.4, 3.5) ---
    section("4. TRAIN / VALIDATION / TEST SPLIT")
    train, val, test = stratified_split(present)
    print(f"Train: {len(train)}   Val: {len(val)}   Test: {len(test)}")
    for name, subset in (("Train", train), ("Val", val), ("Test", test)):
        props = subset["class_label"].value_counts(normalize=True)
        summary = ", ".join(f"{c}={props.get(c, 0):.2%}" for c in dist)
        print(f"  {name:<6} proportions -> {summary}")

    section("DONE")
    print(f"Figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
