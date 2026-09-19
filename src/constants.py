"""Canonical class definitions shared across the pipeline.

The RSNA ``stage_2_detailed_class_info.csv`` file uses three raw label
strings. We keep the canonical class order fixed everywhere (data layer,
models, inference) so that a model's output index always maps to the same
class name.

Raw CSV labels
--------------
- ``Normal``
- ``Lung Opacity``               -> pneumonia-positive (opacity present)
- ``No Lung Opacity / Not Normal`` -> abnormal but not pneumonia
"""

from __future__ import annotations

# Canonical class order. Index position is the model output index.
CLASSES: list[str] = [
    "Normal",
    "Lung Opacity",
    "No Lung Opacity / Not Normal",
]

# Map any accepted raw label spelling to a canonical class in CLASSES.
# The design document refers to "Lung Opacity (Pneumonia)"; the raw dataset
# uses "Lung Opacity". Both map to the same canonical class.
RAW_TO_CANONICAL: dict[str, str] = {
    "Normal": "Normal",
    "Lung Opacity": "Lung Opacity",
    "Lung Opacity (Pneumonia)": "Lung Opacity",
    "No Lung Opacity / Not Normal": "No Lung Opacity / Not Normal",
}

# Convenience lookups.
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
INDEX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CLASSES)}

NUM_CLASSES: int = len(CLASSES)
