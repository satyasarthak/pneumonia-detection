"""Train transfer-learning models and select the best across all models.

Trains one or more pretrained backbones (default: MobileNetV2, ResNet50) with
custom 3-class heads, evaluates them plus the existing baseline on the SAME
held-out test set, compares by macro-F1, and serializes the winner to
models/best_model.keras.

CPU training is slow, so image count / epochs / size are configurable. Use
--sample-per-class 0 for the full dataset.

Usage (project venv):
    C:\\venvs\\pneu\\Scripts\\python.exe run_train_transfer.py \
        --backbones MobileNetV2 ResNet50 --sample-per-class 500 --epochs 5 --image-size 160
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.evaluate import commentary, compare_models, evaluate_model, select_best
from src.generator import XrayBatchGenerator
from src.images import resolve_images
from src.labels import deduplicate_patients, load_labels, restrict_labels
from src.registry import save_model
from src.split import stratified_split
from src.train import train_transfer

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_INFO_CSV = os.path.join(HERE, "stage_2_detailed_class_info.csv")
TRAIN_ZIP = os.path.join(HERE, "stage_2_train_images.zip")
OUT_DIR = os.path.join(HERE, "outputs")
MODEL_DIR = os.path.join(HERE, "models")


def _subsample(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    if per_class <= 0:
        return df
    parts = [
        g.sample(n=min(per_class, len(g)), random_state=seed)
        for _, g in df.groupby("class_label")
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _plot_confusion(cm, title, path):
    import matplotlib.pyplot as plt

    from src.constants import CLASSES

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["MobileNetV2", "ResNet50"])
    ap.add_argument("--sample-per-class", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from tensorflow import keras

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    size = (args.image_size, args.image_size)

    print("Preparing data...")
    clean = restrict_labels(deduplicate_patients(load_labels(CLASS_INFO_CSV, verbose=False)))
    present, _ = resolve_images(clean, TRAIN_ZIP)
    present = _subsample(present, args.sample_per_class)
    train_df, val_df, test_df = stratified_split(present)
    test_df.to_csv(os.path.join(MODEL_DIR, "test_split.csv"), index=False)
    print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}  image_size={size}")

    results: dict[str, dict] = {}
    trained: dict[str, "keras.Model"] = {}

    # Train each transfer backbone.
    for backbone in args.backbones:
        print(f"\n=== Training transfer model: {backbone} ===")
        model, _ = train_transfer(
            backbone, train_df, val_df, TRAIN_ZIP,
            image_size=size, batch_size=args.batch_size, epochs=args.epochs,
        )
        path = os.path.join(MODEL_DIR, f"transfer_{backbone.lower()}.keras")
        model.save(path)
        trained[backbone] = model

        test_gen = XrayBatchGenerator(
            test_df, TRAIN_ZIP, batch_size=args.batch_size,
            target_channels=3, image_size=size, shuffle=False,
        )
        m = evaluate_model(model, test_gen, model_name=backbone)
        results[backbone] = m
        print("  " + commentary(m))
        _plot_confusion(m["confusion_matrix"], f"Confusion - {backbone}",
                        os.path.join(OUT_DIR, f"confusion_transfer_{backbone.lower()}.png"))

    # Include the existing baseline in the comparison if present.
    baseline_path = os.path.join(MODEL_DIR, "baseline_cnn.keras")
    if os.path.exists(baseline_path):
        print("\n=== Evaluating existing Baseline CNN ===")
        baseline = keras.models.load_model(baseline_path)
        # Baseline was trained at 128x128 grayscale; evaluate at its own geometry.
        test_gen = XrayBatchGenerator(
            test_df, TRAIN_ZIP, batch_size=args.batch_size,
            target_channels=1, image_size=(128, 128), shuffle=False,
        )
        m = evaluate_model(baseline, test_gen, model_name="Baseline CNN")
        results["Baseline CNN"] = m
        trained["Baseline CNN"] = baseline
        print("  " + commentary(m))

    # Compare and select the best.
    print("\n=== Model comparison ===")
    table = compare_models(results)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

    best_name, rationale = select_best(table, metric="macro_f1")
    print(f"\nBest model: {best_name}")
    print(f"Rationale: {rationale}")

    # Serialize the winner as the served model.
    best_path = save_model(trained[best_name], os.path.join(MODEL_DIR, "best_model.keras"))
    print(f"Serialized best model -> {best_path}")

    # Record the winner's input geometry so the app/Docker can be configured.
    geom = "1,128,128" if best_name == "Baseline CNN" else f"3,{size[0]},{size[1]}"
    with open(os.path.join(MODEL_DIR, "best_model_geometry.txt"), "w") as fh:
        fh.write(geom)
    print(f"Best model geometry (channels,H,W): {geom}")


if __name__ == "__main__":
    main()
