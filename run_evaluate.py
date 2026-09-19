"""Evaluate trained model(s) on the held-out test split and build a comparison.

Loads the saved baseline model and its persisted test split, computes
multi-class metrics, prints per-class detail and a comparison table, selects
the best model by macro-F1, and writes a confusion-matrix figure.

As more models are trained (transfer models in Task 11+), add them to
MODELS below and they join the comparison automatically.

Usage:
    C:\\venvs\\pneu\\Scripts\\python.exe run_evaluate.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.evaluate import commentary, compare_models, evaluate_model, select_best
from src.generator import XrayBatchGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_ZIP = os.path.join(HERE, "stage_2_train_images.zip")
OUT_DIR = os.path.join(HERE, "outputs")
MODEL_DIR = os.path.join(HERE, "models")

# (display_name, model_file, target_channels, image_size)
MODELS = [
    ("Baseline CNN", "baseline_cnn.keras", 1, (128, 128)),
    ("MobileNetV2", "transfer_mobilenetv2.keras", 3, (160, 160)),
    ("ResNet50", "transfer_resnet50.keras", 3, (160, 160)),
]


def _plot_confusion(cm, title, path):
    import matplotlib.pyplot as plt
    import numpy as np

    from src.constants import CLASSES

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    from tensorflow import keras

    os.makedirs(OUT_DIR, exist_ok=True)
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_split.csv"), dtype=str)
    print(f"Test set: {len(test_df)} images")

    results: dict[str, dict] = {}
    for name, fname, channels, size in MODELS:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"  (skip) {name}: {fname} not found")
            continue
        print(f"Evaluating {name} ...")
        model = keras.models.load_model(path)
        test_gen = XrayBatchGenerator(
            test_df, TRAIN_ZIP, batch_size=32,
            target_channels=channels, image_size=size, shuffle=False,
        )
        m = evaluate_model(model, test_gen, model_name=name)
        results[name] = m

        print("  " + commentary(m))
        print("  Per-class:")
        for cls, d in m["per_class"].items():
            print(f"    {cls:<32} P={d['precision']:.3f} R={d['recall']:.3f} "
                  f"F1={d['f1']:.3f} (n={d['support']})")
        _plot_confusion(
            m["confusion_matrix"], f"Confusion Matrix - {name}",
            os.path.join(OUT_DIR, f"confusion_{fname.replace('.keras', '')}.png"),
        )

    if not results:
        print("No models evaluated.")
        return

    print("\n=== Model comparison ===")
    table = compare_models(results)
    print(table.to_string(index=False))

    best_name, rationale = select_best(table, metric="macro_f1")
    print(f"\nBest model: {best_name}")
    print(f"Rationale: {rationale}")


if __name__ == "__main__":
    main()
