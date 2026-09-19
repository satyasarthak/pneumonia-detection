"""Tests for the label layer.

Feature: pneumonia-detection
Covers Properties 1, 2, 3 and the record-count reporting example (Req 1.2).
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.constants import CLASSES
from src.labels import deduplicate_patients, load_labels, restrict_labels

_ALL_LABELS = CLASSES + ["Bogus", "", "random-junk", "Lung Opacity (Pneumonia)"]


@st.composite
def label_frames(draw):
    """Generate synthetic [patientId, class_label] frames."""
    size = draw(st.integers(min_value=0, max_value=40))
    pids = draw(
        st.lists(
            st.sampled_from([f"p{i}" for i in range(15)]),
            min_size=size,
            max_size=size,
        )
    )
    labels = draw(
        st.lists(st.sampled_from(_ALL_LABELS), min_size=size, max_size=size)
    )
    return pd.DataFrame({"patientId": pids, "class_label": labels})


# --- Property 1: Label parsing produces one labeled record per source row ---
# patientIds in the RSNA data are hex UUID strings, so restrict generated IDs
# to a realistic identifier alphabet (letters, digits, hyphen). This avoids
# pathological CSV round-trip artifacts (null bytes, whitespace-only) that do
# not occur in the real dataset while still exercising arbitrary IDs.
_id_text = st.text(
    alphabet="abcdefABCDEF0123456789-", min_size=1, max_size=12
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    rows=st.lists(
        st.tuples(_id_text, st.sampled_from(CLASSES)),
        min_size=1,
        max_size=30,
    )
)
def test_property_1_load_labels_one_record_per_row(rows, tmp_path):
    """Feature: pneumonia-detection, Property 1: Label parsing produces one
    labeled record per source row. Validates Requirements 1.1."""
    csv = tmp_path / "labels.csv"
    df_in = pd.DataFrame(rows, columns=["patientId", "class"])
    df_in.to_csv(csv, index=False)

    out = load_labels(str(csv), verbose=False)

    assert len(out) == len(df_in)
    assert list(out.columns) == ["patientId", "class_label"]
    assert (out["patientId"].str.len() > 0).all()
    assert out["class_label"].notna().all()


# --- Property 2: De-duplication yields unique patients ---
@settings(max_examples=100)
@given(df=label_frames())
def test_property_2_deduplicate_unique_patients(df):
    """Feature: pneumonia-detection, Property 2: De-duplication yields unique
    patients. Validates Requirements 1.3."""
    out = deduplicate_patients(df)

    assert out["patientId"].is_unique
    assert len(out) <= len(df)
    assert set(out["patientId"]).issubset(set(df["patientId"]))


# --- Property 3: Labels are restricted to three categories ---
@settings(max_examples=100)
@given(df=label_frames())
def test_property_3_restrict_labels(df):
    """Feature: pneumonia-detection, Property 3: Labels are restricted to three
    categories. Validates Requirements 1.4."""
    out = restrict_labels(df)
    assert set(out["class_label"]).issubset(set(CLASSES))


# --- Example: record-count reporting (Req 1.2) ---
def test_load_labels_reports_shape(raw_label_df, tmp_path, capsys):
    csv = tmp_path / "labels.csv"
    raw_label_df.rename(columns={"class_label": "class"}).to_csv(csv, index=False)

    out = load_labels(str(csv), verbose=True)
    captured = capsys.readouterr().out

    assert f"records: {len(out)}" in captured
    assert str(out.shape) in captured


def test_deduplicate_collapses_known_duplicate(raw_label_df):
    out = deduplicate_patients(raw_label_df)
    # p1 appears twice in the fixture -> one row after dedup.
    assert (out["patientId"] == "p1").sum() == 1
    assert out["patientId"].is_unique


def test_restrict_drops_bogus_label(raw_label_df):
    out = restrict_labels(raw_label_df)
    assert "Bogus" not in set(out["class_label"])
