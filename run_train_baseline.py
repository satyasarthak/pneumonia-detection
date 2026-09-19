"""Train the from-scratch baseline CNN (interim 'Model Building' deliverable).

Loads labels, resolves images, splits stratified, then trains the baseline CNN
with class weights for the ~2x imbalance. Saves the trained model and a
training-history plot into ``outputs/``.

Because CPU training on all ~26k images is slow, the script exposes
``--sample-per-class``, ``--epochs``, ``--image-size``, and ``--batch-size`` so
a bounded run can demonstrate the full pipeline end to end. Use
``--sample-per-class 0`` to train on the entire dataset.

Usage (from the project venv):
    C:\\venvs\\pneu\\Scripts\\python.exe run_train_baseline.py --sample-per-class 400 --epochs 5
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.images import resolve_images
from src.labels import deduplicate_patients, load_labels, restrict_labels
from src.split import stratified_split
from src.train import train_baseline

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-per-class", type=int, default=400,
                    help="Images per class to use (0 = full dataset).")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading and preparing labels...")
    clean = restrict_labels(deduplicate_patients(load_labels(CLASS_INFO_CSV, verbose=False)))
    present, _ = resolve_images(clean, TRAIN_ZIP)
    present = _subsample(present, args.sample_per_class)
    print(f"Training on {len(present)} images "
          f"({'full dataset' if args.sample_per_class == 0 else str(args.sample_per_class) + '/class'})")

    train_df, val_df, test_df = stratified_split(present)
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    size = (args.image_size, args.image_size)
    print(f"Training baseline CNN  image_size={size}  epochs={args.epochs} ...")
    model, history = train_baseline(
        train_df, val_df, TRAIN_ZIP,
        image_size=size, batch_size=args.batch_size, epochs=args.epochs,
    )

    # Persist model + split so evaluation (Task 11) can reuse the same test set.
    model_path = os.path.join(MODEL_DIR, "baseline_cnn.keras")
    model.save(model_path)
    test_df.to_csv(os.path.join(MODEL_DIR, "test_split.csv"), index=False)
    print(f"Saved model -> {model_path}")

    # Training-history figure.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h = history.history
    axes[0].plot(h.get("loss", []), label="train")
    axes[0].plot(h.get("val_loss", []), label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(h.get("accuracy", []), label="train")
    axes[1].plot(h.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "baseline_training_history.png"), dpi=120)
    print("Saved outputs/baseline_training_history.png")

    final = {k: (v[-1] if v else None) for k, v in h.items()}
    print("\nFinal epoch metrics:")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: n/a")


if __name__ == "__main__":
    main()
