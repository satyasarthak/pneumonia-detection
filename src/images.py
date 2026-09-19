"""Image resolution and DICOM decoding.

Resolves which label records have an available ``.dcm`` file and decodes a
DICOM file into a 2D pixel array. An image source may be either a directory
of ``<patientId>.dcm`` files or a ``.zip`` archive containing them (as the
RSNA data ships).

Requirements: 1.5, 3.1
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImageSource:
    """Uniform accessor over a directory or a zip archive of DICOM files.

    Files are looked up by ``patientId`` -> ``<patientId>.dcm``. Inside a zip
    the entry may be nested (e.g. ``stage_2_train_images/<id>.dcm``); the
    basename is used to build an index so nesting and macOS ``__MACOSX``
    sidecar entries are handled transparently.
    """

    def __init__(self, path: str):
        self.path = path
        self._is_zip = zipfile.is_zipfile(path) if os.path.isfile(path) else False
        self._zip: zipfile.ZipFile | None = None
        self._zip_index: dict[str, str] = {}

        if self._is_zip:
            self._zip = zipfile.ZipFile(path)
            for name in self._zip.namelist():
                base = os.path.basename(name)
                # Skip directory entries and macOS resource-fork sidecars.
                if not base.endswith(".dcm") or base.startswith("._"):
                    continue
                self._zip_index[base] = name
        elif not os.path.isdir(path):
            raise FileNotFoundError(
                f"Image source is neither a directory nor a zip: {path}"
            )

    def _entry_name(self, patient_id: str) -> str:
        return f"{patient_id}.dcm"

    def exists(self, patient_id: str) -> bool:
        """Return True if a DICOM file for ``patient_id`` is available."""
        entry = self._entry_name(patient_id)
        if self._is_zip:
            return entry in self._zip_index
        return os.path.isfile(os.path.join(self.path, entry))

    def read_bytes(self, patient_id: str) -> bytes:
        """Return the raw DICOM bytes for ``patient_id``."""
        entry = self._entry_name(patient_id)
        if self._is_zip:
            assert self._zip is not None
            return self._zip.read(self._zip_index[entry])
        with open(os.path.join(self.path, entry), "rb") as fh:
            return fh.read()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def resolve_images(
    df: pd.DataFrame, image_source: Union[str, ImageSource]
) -> tuple[pd.DataFrame, list[str]]:
    """Partition records into those with an available image and those without.

    Parameters
    ----------
    df:
        Records with a ``patientId`` column.
    image_source:
        A path (directory or zip) or an :class:`ImageSource`.

    Returns
    -------
    (present_df, missing_patient_ids)
        ``present_df`` keeps only rows whose image exists; the missing IDs are
        logged and returned so they can be reported and excluded downstream.

    Requirements: 1.5
    """
    owns_source = isinstance(image_source, str)
    source = ImageSource(image_source) if owns_source else image_source

    try:
        present_mask = df["patientId"].map(source.exists)
    finally:
        if owns_source:
            source.close()

    present_df = df.loc[present_mask].reset_index(drop=True)
    missing_ids = df.loc[~present_mask, "patientId"].astype(str).tolist()

    if missing_ids:
        logger.warning(
            "resolve_images: %d record(s) have no image and were excluded",
            len(missing_ids),
        )

    return present_df, missing_ids


def load_dicom(
    source: Union[str, bytes, ImageSource], patient_id: str | None = None
) -> np.ndarray:
    """Decode a DICOM into a 2D pixel array.

    ``source`` may be:
      * a filesystem path to a ``.dcm`` file,
      * raw DICOM ``bytes``,
      * an :class:`ImageSource` together with ``patient_id``.

    Decode failures are logged and re-raised as :class:`DicomDecodeError` so
    callers can treat a corrupt file like a missing one.

    Requirements: 3.1
    """
    import pydicom

    try:
        if isinstance(source, ImageSource):
            if patient_id is None:
                raise ValueError("patient_id is required when source is ImageSource")
            data = source.read_bytes(patient_id)
            ds = pydicom.dcmread(io.BytesIO(data))
        elif isinstance(source, bytes):
            ds = pydicom.dcmread(io.BytesIO(source))
        else:
            ds = pydicom.dcmread(source)
        arr = ds.pixel_array
    except Exception as exc:  # noqa: BLE001 - normalize to a domain error
        logger.warning("load_dicom: failed to decode (%s): %s", patient_id, exc)
        raise DicomDecodeError(str(exc)) from exc

    return np.asarray(arr)


class DicomDecodeError(Exception):
    """Raised when a DICOM file cannot be decoded into a pixel array."""
