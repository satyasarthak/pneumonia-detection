"""Label loading, de-duplication, and restriction.

Implements the first stage of the Data_Pipeline: parse the RSNA
``stage_2_detailed_class_info.csv`` into a clean ``[patientId, class_label]``
frame, collapse duplicate patients to a single label, and drop any label
outside the three canonical categories.

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from __future__ import annotations

import pandas as pd

from .constants import CLASSES, RAW_TO_CANONICAL


def load_labels(class_info_csv: str, *, verbose: bool = True) -> pd.DataFrame:
    """Parse the Label_Source into columns ``[patientId, class_label]``.

    Each source row becomes exactly one record. ``class`` values are mapped
    to their canonical spelling where a mapping exists; unmapped values are
    preserved verbatim so that :func:`restrict_labels` can drop them later.

    Parameters
    ----------
    class_info_csv:
        Path to ``stage_2_detailed_class_info.csv``.
    verbose:
        When ``True`` (default), print the record count and table shape,
        satisfying the data-overview reporting requirement.

    Returns
    -------
    pandas.DataFrame
        Columns ``patientId`` (str) and ``class_label`` (str), one row per
        source row.

    Requirements: 1.1, 1.2
    """
    # Read identifier/label columns as raw strings. Disabling NA-coercion
    # (keep_default_na=False) stops pandas from turning legitimate values like
    # "NULL", "NA", or "None" into NaN, which would corrupt patient IDs.
    df = pd.read_csv(class_info_csv, dtype=str, keep_default_na=False, na_values=[])

    if "patientId" not in df.columns or "class" not in df.columns:
        raise ValueError(
            "Label_Source must contain 'patientId' and 'class' columns; "
            f"found {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "patientId": df["patientId"].astype(str),
            "class_label": df["class"].astype(str).map(
                lambda v: RAW_TO_CANONICAL.get(v, v)
            ),
        }
    )

    if verbose:
        print(f"[load_labels] records: {len(out)}  shape: {out.shape}")

    return out


def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the frame to one row per ``patientId``.

    RSNA lists one bounding-box row per opacity, so a pneumonia-positive
    patient can appear multiple times. For classification we need a single
    label per patient. The first occurrence's label is retained; because all
    rows for a given patient share the same detailed class, this is stable.

    Requirements: 1.3
    """
    if df.empty:
        return df.copy()

    deduped = (
        df.drop_duplicates(subset="patientId", keep="first")
        .reset_index(drop=True)
    )
    return deduped


def restrict_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose ``class_label`` is one of the canonical classes.

    Any row bearing a label outside :data:`src.constants.CLASSES` is dropped.

    Requirements: 1.4
    """
    mask = df["class_label"].isin(set(CLASSES))
    return df.loc[mask].reset_index(drop=True)
