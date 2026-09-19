"""Smoke tests for deployment artifacts.

Feature: pneumonia-detection
Verifies the packaging deliverables exist and declare what the app needs
(Req 6.4, 6.5, 6.6). These are file/config checks, not a container build.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path: str) -> str:
    with open(os.path.join(HERE, path), encoding="utf-8") as fh:
        return fh.read()


# --- Req 6.4: requirements.txt declares backend + frontend deps ---
def test_requirements_declares_dependencies():
    reqs = _read("requirements.txt").lower()
    for pkg in ("streamlit", "tensorflow", "pydicom", "numpy", "pandas", "scikit-learn", "pillow"):
        assert pkg in reqs, f"missing dependency: {pkg}"


# --- Req 6.5: Dockerfile builds/parses and runs the app ---
def test_dockerfile_present_and_valid():
    df = _read("Dockerfile")
    # A FROM instruction is present (comments may precede it).
    assert any(line.strip().startswith("FROM ") for line in df.splitlines())
    assert "requirements.txt" in df
    assert "pip install" in df
    assert "EXPOSE 8501" in df
    assert "streamlit" in df and "run" in df
    assert "app.py" in df


def test_devcontainer_forwards_streamlit_port():
    dc = _read(os.path.join(".devcontainer", "devcontainer.json"))
    assert "8501" in dc
    assert "requirements.txt" in dc


# --- Req 6.6: repository contains app + model artifact ---
def test_repository_contains_app_and_model():
    assert os.path.exists(os.path.join(HERE, "app.py"))
    assert os.path.isdir(os.path.join(HERE, "src"))
    # The served model artifact is present (serialized in Task 12).
    assert os.path.exists(os.path.join(HERE, "models", "best_model.keras"))
