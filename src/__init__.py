"""Pneumonia detection package.

End-to-end pipeline for classifying chest X-ray radiographs from the RSNA
Pneumonia Detection dataset into three categories.
"""

from .constants import CLASSES, RAW_TO_CANONICAL

__all__ = ["CLASSES", "RAW_TO_CANONICAL"]
