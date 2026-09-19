"""Model evaluation, comparison, and selection.

Computes multi-class metrics on the held-out test subset, assembles a
comparison table across models, and selects the best performer by macro-F1
(robust to the class imbalance in this dataset).

scikit-learn is used for metrics; TensorFlow is only needed when evaluating a
Keras model against a generator (imported lazily via the generator path).

Requirements: 4.3, 4.4, 5.3, 5.4, 5.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .constants import CLASSES, NUM_CLASSES


def compute_metrics(y_true, y_pred) -> dict:
    """Compute multi-class metrics from integer label arrays.

    Returns accuracy, macro precision/recall/F1, a per-class report, and the
    3x3 confusion matrix (labels fixed to the canonical class order so the
    matrix is comparable across models).

    Requirements: 4.3, 5.3
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(NUM_CLASSES))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        CLASSES[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(NUM_CLASSES)
    }

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _predict_over_generator(model, gen) -> tuple[np.ndarray, np.ndarray]:
    """Run a Keras model over a generator, returning (y_true, y_pred) indices."""
    y_true: list[int] = []
    y_pred: list[int] = []
    for i in range(len(gen)):
        x, y = gen[i]
        if x.shape[0] == 0:
            continue
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(y, axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def evaluate_model(model, test_gen, model_name: str = "model") -> dict:
    """Evaluate a trained Keras model on a test generator.

    Returns the metrics dict from :func:`compute_metrics` plus ``model_name``.

    Requirements: 4.3, 5.3
    """
    y_true, y_pred = _predict_over_generator(model, test_gen)
    metrics = compute_metrics(y_true, y_pred)
    metrics["model_name"] = model_name
    return metrics


def commentary(metrics: dict) -> str:
    """One-line interpretive commentary on a model's performance.

    Requirements: 4.4
    """
    acc = metrics["accuracy"]
    f1 = metrics["macro_f1"]
    # Identify the weakest class by recall.
    weakest = min(metrics["per_class"].items(), key=lambda kv: kv[1]["recall"])
    return (
        f"{metrics.get('model_name', 'model')}: accuracy={acc:.3f}, "
        f"macro-F1={f1:.3f}. Weakest class by recall is '{weakest[0]}' "
        f"({weakest[1]['recall']:.3f})."
    )


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison table: one row per model, identical metric columns.

    Requirements: 4.4, 5.4
    """
    cols = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
            }
        )
    return pd.DataFrame(rows, columns=cols)


def select_best(comparison: pd.DataFrame, metric: str = "macro_f1") -> tuple[str, str]:
    """Select the best model by ``metric`` (argmax) and return a rationale.

    Requirements: 5.5
    """
    if comparison.empty:
        raise ValueError("comparison table is empty")

    best_idx = comparison[metric].idxmax()
    best_row = comparison.loc[best_idx]
    best_name = str(best_row["model"])
    best_val = float(best_row[metric])

    others = comparison.drop(index=best_idx)
    if others.empty:
        rationale = (
            f"'{best_name}' selected as the only model, with {metric}={best_val:.3f}."
        )
    else:
        runner = others.loc[others[metric].idxmax()]
        rationale = (
            f"'{best_name}' selected: highest {metric} ({best_val:.3f}), "
            f"ahead of '{runner['model']}' ({float(runner[metric]):.3f}). "
            f"Macro-F1 is used because it weights the three classes equally "
            f"despite the dataset's class imbalance."
        )
    return best_name, rationale
