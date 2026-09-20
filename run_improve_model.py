"""Train an improved MobileNetV2 using the EDA-driven levers and measure gain.

Applies the improvements the EDA motivated:
  - Fine-tuning the backbone (two-phase) instead of freezing it
  - Data augmentation (rotation/zoom/flip/brightness)
  - Higher resolution (224 vs 160) and more data/epochs
  - Class weights for the ~2x imbalance

Evaluates on the same held-out test set and compares against the existing
frozen-base MobileNetV2. If the improved model wins on macro-F1, it becomes the
new best_model.keras.

Usage:
    C:\\venvs\\pneu\\Scripts\\python.exe run_improve_model.py \
        --sample-per-class 800 --image-size 224 --head-epochs 4 --finetune-epochs 6
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
from src.train import train_transfer_finetune

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_INFO_CSV = os.path.join(HERE, "stage_2_detailed_class_info.csv")
TRAIN_ZIP = os.path.join(HERE, "stage_2_train_images.zip")
OUT_DIR = os.path.join(HERE, "outputs")
MODEL_DIR = os.path.join(HERE, "models")


def _subsample(df, per_class, seed=42):
    if per_class <= 0:
        return df
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed)
             for _, g in df.groupby("class_label")]
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
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-per-class", type=int, default=800)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--head-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from tensorflow import keras

    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
    size = (args.image_size, args.image_size)

    print("Preparing data...")
    clean = restrict_labels(deduplicate_patients(load_labels(CLASS_INFO_CSV, verbose=False)))
    present, _ = resolve_images(clean, TRAIN_ZIP)
    present = _subsample(present, args.sample_per_class)
    train_df, val_df, test_df = stratified_split(present)
    print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)} image_size={size}")

    print("\n=== Training improved MobileNetV2 (fine-tuned + augmented) ===")
    model, hist = train_transfer_finetune(
        "MobileNetV2", train_df, val_df, TRAIN_ZIP,
        image_size=size, batch_size=args.batch_size,
        head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs,
    )
    improved_path = os.path.join(MODEL_DIR, "mobilenetv2_finetuned.keras")
    model.save(improved_path)

    results = {}
    test_gen = XrayBatchGenerator(test_df, TRAIN_ZIP, batch_size=args.batch_size,
                                  target_channels=3, image_size=size, shuffle=False)
    m = evaluate_model(model, test_gen, model_name="MobileNetV2 (fine-tuned)")
    results["MobileNetV2 (fine-tuned)"] = m
    print("  " + commentary(m))
    _plot_confusion(m["confusion_matrix"], "Confusion - MobileNetV2 (fine-tuned)",
                    os.path.join(OUT_DIR, "confusion_mobilenetv2_finetuned.png"))

    # Compare against the previous frozen-base MobileNetV2 if available.
    prev = os.path.join(MODEL_DIR, "transfer_mobilenetv2.keras")
    if os.path.exists(prev):
        print("\n=== Evaluating previous frozen-base MobileNetV2 (160x160) ===")
        prev_model = keras.models.load_model(prev)
        prev_gen = XrayBatchGenerator(test_df, TRAIN_ZIP, batch_size=args.batch_size,
                                      target_channels=3, image_size=(160, 160), shuffle=False)
        pm = evaluate_model(prev_model, prev_gen, model_name="MobileNetV2 (frozen)")
        results["MobileNetV2 (frozen)"] = pm
        print("  " + commentary(pm))

    print("\n=== Comparison ===")
    table = compare_models(results)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(OUT_DIR, "model_comparison_improved.csv"), index=False)

    best_name, rationale = select_best(table, metric="macro_f1")
    print(f"\nBest: {best_name}\n{rationale}")

    # Promote the improved model to best_model if it wins.
    if best_name == "MobileNetV2 (fine-tuned)":
        save_model(model, os.path.join(MODEL_DIR, "best_model.keras"))
        with open(os.path.join(MODEL_DIR, "best_model_geometry.txt"), "w") as fh:
            fh.write(f"3,{size[0]},{size[1]}")
        print(f"Promoted fine-tuned model to best_model.keras (geometry 3,{size[0]},{size[1]})")


if __name__ == "__main__":
    main()
