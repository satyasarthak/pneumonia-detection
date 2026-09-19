# Implementation Plan: Pneumonia Detection

## Overview

This plan converts the design into incremental, code-focused steps. It builds bottom-up: project scaffold → pure data-layer functions → preprocessing and generators → EDA → baseline CNN → transfer models → evaluation/selection/registry → shared inference → Streamlit app → deployment → business report. Each step wires into the previous one so there is no orphaned code. Property tests (Properties 1–18 from the design) are placed next to the code they validate; example, integration, and smoke tests cover display, wiring, and infrastructure criteria.

Implementation language: **Python** (as specified in the design).

## Tasks

- [x] 1. Set up project scaffold and dependencies
  - Create `src/` package with `__init__.py` and module files: `labels.py`, `images.py`, `preprocess.py`, `split.py`, `generator.py`, `eda.py`, `models.py`, `evaluate.py`, `registry.py`, `inference.py`
  - Create `tests/` directory with `__init__.py` and a `conftest.py` providing shared fixtures (tiny synthetic DataFrames, in-memory pixel arrays)
  - Define the canonical `CLASSES` constant (Normal, Lung Opacity (Pneumonia), No Lung Opacity / Not Normal) in a shared module imported everywhere
  - Create root `requirements.txt` declaring backend + frontend deps: streamlit, tensorflow, pydicom, numpy, pandas, pillow, scikit-learn, matplotlib, seaborn, hypothesis, pytest
  - _Requirements: 3.2, 5.1, 6.4_

- [x] 2. Implement label loading, de-duplication, and label restriction
  - [x] 2.1 Implement `load_labels`, `deduplicate_patients`, `restrict_labels` in `labels.py`
    - `load_labels` parses `stage_2_detailed_class_info.csv` into `[patientId, class_label]` and reports row/column counts
    - `deduplicate_patients` collapses to one row per `patientId`
    - `restrict_labels` drops any row whose label is outside the three canonical categories
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x]* 2.2 Write property test for label parsing
    - **Property 1: Label parsing produces one labeled record per source row**
    - **Validates: Requirements 1.1**

  - [x]* 2.3 Write property test for de-duplication
    - **Property 2: De-duplication yields unique patients**
    - **Validates: Requirements 1.3**

  - [x]* 2.4 Write property test for label restriction
    - **Property 3: Labels are restricted to three categories**
    - **Validates: Requirements 1.4**

  - [x]* 2.5 Write example test for record-count reporting
    - Assert `load_labels` reports number of records and table dimensions on a small fixture
    - _Requirements: 1.2_

- [x] 3. Implement image resolution and DICOM decode
  - [x] 3.1 Implement `resolve_images` and `load_dicom` in `images.py`
    - `resolve_images` partitions records into `(present_df, missing_patient_ids)`, logging and excluding missing files
    - `load_dicom` decodes a `.dcm` file into a 2D pixel array via `pydicom`, catching/logging decode failures and treating them as missing
    - _Requirements: 1.5, 3.1_

  - [x]* 3.2 Write property test for image resolution partition
    - **Property 4: Missing images partition the records**
    - **Validates: Requirements 1.5**

  - [x]* 3.3 Write integration test for DICOM decode
    - Decode a real `.dcm` fixture and assert a 2D numeric array is returned
    - _Requirements: 3.1_

- [x] 4. Implement preprocessing transforms
  - [x] 4.1 Implement `to_channels`, `normalize`, `resize` in `preprocess.py`
    - `to_channels` converts between grayscale (1ch) and RGB (3ch) via channel replication
    - `normalize` scales pixel values into [0, 1] preserving relative ordering
    - `resize` resizes to the model input resolution (default 224×224)
    - _Requirements: 3.2, 3.6_

  - [x]* 4.2 Write property test for channel conversion
    - **Property 6: Channel conversion produces the target format**
    - **Validates: Requirements 3.2**

  - [x]* 4.3 Write property test for normalization
    - **Property 7: Normalization bounds pixel values**
    - **Validates: Requirements 3.6**

- [x] 5. Implement stratified splitting
  - [x] 5.1 Implement `stratified_split` in `split.py`
    - Partition de-duplicated records into train/val/test (default 0.7/0.15/0.15) preserving class proportions using a fixed seed
    - _Requirements: 3.4, 3.5_

  - [x]* 5.2 Write property test for disjoint/covering partition
    - **Property 8: Dataset partition is disjoint and covering**
    - **Validates: Requirements 3.4**

  - [x]* 5.3 Write property test for stratification
    - **Property 9: Stratified split preserves class proportions**
    - **Validates: Requirements 3.5**

- [x] 6. Implement the batched Keras generator
  - [x] 6.1 Implement `XrayBatchGenerator` in `generator.py`
    - Subclass `keras.utils.Sequence`; load only the current batch's images incrementally
    - Apply resize → normalize → channel-convert per sample; stack into `(batch, H, W, C)`
    - Support optional on-the-fly augmentation and configurable target channels
    - _Requirements: 3.7, 3.2, 3.6_

  - [x]* 6.2 Write property test for generator epoch coverage
    - **Property 10: Batched generator covers the dataset exactly once per epoch**
    - **Validates: Requirements 3.7**

- [x] 7. Checkpoint - Ensure all data-layer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement EDA module
  - [x] 8.1 Implement `sample_images_per_class`, `class_distribution`, `imbalance_report` in `eda.py`
    - Randomly sample n images per class and annotate each with its Class_Label for display
    - Compute per-class `patientId` counts and render distribution
    - Flag imbalance when max/min class ratio exceeds threshold and record observations
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x]* 8.2 Write property test for class distribution aggregation
    - **Property 5: Class distribution aggregation is exact**
    - **Validates: Requirements 2.3**

  - [x]* 8.3 Write example tests for EDA display and imbalance flag
    - Assert sample grid returns n images per class annotated with labels (2.1, 2.2)
    - Assert imbalance flag is set on a deliberately skewed fixture (2.4)
    - Assert a before/after preprocessing pair is produced for a sample image (3.3)
    - _Requirements: 2.1, 2.2, 2.4, 3.3_

- [x] 9. Implement modeling layer
  - [x] 9.1 Implement `build_baseline_cnn` in `models.py`
    - CNN from scratch: 3 Conv/BN/ReLU/MaxPool blocks → GAP → Dropout → Dense(128) → Dense(3, softmax)
    - Input shape (224, 224, 1); compile with categorical cross-entropy and `class_weight` support
    - _Requirements: 4.1_

  - [x] 9.2 Implement `build_transfer_model` in `models.py`
    - Pretrained backbone (VGG16 | ResNet50 | MobileNetV2, `include_top=False`, imagenet weights) + GAP → Dropout → Dense(128) → Dense(3, softmax)
    - Input shape (224, 224, 3); build at least two distinct backbones (default ResNet50 + MobileNetV2)
    - _Requirements: 5.1, 5.2_

  - [x]* 9.3 Write property test for 3-class softmax output
    - **Property 11: Classification models produce a valid 3-class softmax output**
    - **Validates: Requirements 4.1, 5.1, 5.2**
    - Use tiny in-memory models and generated input batches for speed

  - [x]* 9.4 Write example test for distinct backbones
    - Assert ≥2 distinct backbones are constructed and differ from the baseline
    - _Requirements: 5.1_

- [x] 10. Implement training entry points
  - [x] 10.1 Implement training functions that fit baseline and transfer models
    - Consume train/val subsets via `XrayBatchGenerator`; apply `class_weight` for imbalance; return trained model + history
    - _Requirements: 4.2, 5.1_

  - [x]* 10.2 Write integration test for a short training run
    - Run a 1–2 step `fit` on tiny generators to confirm train/val wiring
    - _Requirements: 4.2_

- [x] 11. Implement evaluation, comparison, and selection
  - [x] 11.1 Implement `evaluate_model`, `compare_models`, `select_best` in `evaluate.py`
    - `evaluate_model` computes accuracy, macro precision/recall/F1, per-class report, confusion matrix on the test subset
    - `compare_models` builds one row per model with identical metric columns; record interpretive commentary for baseline and transfer models
    - `select_best` returns `(best_model_name, rationale)` = argmax over macro-F1
    - _Requirements: 4.3, 4.4, 5.3, 5.4, 5.5_

  - [x]* 11.2 Write property test for multi-class metric validity
    - **Property 12: Multi-class metrics are valid**
    - **Validates: Requirements 4.3, 5.3**

  - [x]* 11.3 Write property test for comparison-table consistency
    - **Property 13: Model comparison table is complete and consistent**
    - **Validates: Requirements 4.4, 5.4**

  - [x]* 11.4 Write property test for best-model selection
    - **Property 14: Best-model selection is the metric optimum**
    - **Validates: Requirements 5.5**

- [x] 12. Implement model registry (serialize / reload)
  - [x] 12.1 Implement `save_model` and `load_model` in `registry.py`
    - Serialize the selected best model to `models/best_model` (SavedModel or `.h5`); reload for inference
    - _Requirements: 5.6, 5.7_

  - [x]* 12.2 Write property test for serialization round-trip
    - **Property 15: Serialization round-trip preserves predictions**
    - **Validates: Requirements 5.6**

- [x] 13. Checkpoint - Ensure modeling and evaluation tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement shared inference service
  - [x] 14.1 Implement `validate_upload`, `preprocess_for_inference`, `predict` in `inference.py`
    - `validate_upload` raises `UnsupportedImageError` for non-DICOM/non-image inputs
    - `preprocess_for_inference` resizes + channel-converts + normalizes for the served model
    - `predict` returns `(predicted_label, {class: probability})` with `label = CLASSES[argmax]`
    - _Requirements: 5.7, 6.1, 6.2, 6.3_

  - [x]* 14.2 Write property test for predicted-label mapping
    - **Property 16: Predicted label maps to a valid category**
    - **Validates: Requirements 5.7, 6.1**

  - [x]* 14.3 Write property test for prediction probability distribution
    - **Property 17: Prediction probabilities are a valid distribution over three classes**
    - **Validates: Requirements 6.2**

  - [x]* 14.4 Write property test for unsupported-upload rejection
    - **Property 18: Unsupported uploads are rejected**
    - **Validates: Requirements 6.3**

- [x] 15. Implement the Streamlit app
  - [x] 15.1 Implement `app.py`
    - Cache the loaded best model with `@st.cache_resource`; provide `st.file_uploader`
    - On upload: `validate_upload` → load DICOM/image → `preprocess_for_inference` → `predict`
    - Display predicted Class_Label and per-class probabilities (bar chart); show error on `UnsupportedImageError`
    - _Requirements: 6.1, 6.2, 6.3_

  - [x]* 15.2 Write smoke test for app wiring
    - Import `app` module and exercise the upload→predict path with a mocked model/fixture
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 16. Implement deployment packaging
  - [x] 16.1 Create `Dockerfile` and finalize `requirements.txt`, add `.devcontainer` for Codespaces
    - Dockerfile: python:3.11-slim base, install requirements, copy app, EXPOSE 8501, run streamlit on 0.0.0.0:8501
    - `.devcontainer` (or documented `streamlit run`) forwards port 8501 to produce a public URL
    - _Requirements: 6.4, 6.5, 6.7_

  - [x]* 16.2 Write smoke tests for deployment artifacts
    - Assert `requirements.txt` exists and declares required deps (6.4)
    - Assert `Dockerfile` builds/parses and the repo contains app + model artifact (6.5, 6.6)
    - _Requirements: 6.4, 6.5, 6.6_

- [x] 17. Implement the Report_Module
  - [x] 17.1 Create the business report deliverable
    - Structure: problem definition, methodology, results, recommendations in non-technical language
    - Include actionable insights from model results + data analysis and clinical decision-support recommendations
    - Embed supporting visualizations (class distribution, confusion matrix, metric comparison charts)
    - _Requirements: 7.1, 7.2, 8.1, 8.2, 8.3_

- [x] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP.
- Each task references specific requirements for traceability; property test tasks reference their design property number.
- Heavy model properties (11, 12, 15, 16) use tiny in-memory models and generated arrays to stay fast and deterministic; each property test runs a minimum of 100 iterations.
- Deployment steps 6.6/6.8 (pushing to a repo and live Codespaces inference) require human action outside a coding agent; the code, Dockerfile, and devcontainer that enable them are produced here, and live-URL verification is a manual integration check.
- Documentation criteria (7.1, 7.2, 8.1–8.3) are satisfied by the Report_Module deliverable and reviewed rather than unit-tested.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "2.4", "2.5", "3.2", "3.3", "4.2", "4.3", "5.1", "8.1", "9.1", "9.2"] },
    { "id": 2, "tasks": ["5.2", "5.3", "6.1", "8.2", "8.3", "9.3", "9.4", "11.1"] },
    { "id": 3, "tasks": ["6.2", "10.1", "11.2", "11.3", "11.4", "14.1"] },
    { "id": 4, "tasks": ["10.2", "12.1", "14.2", "14.3", "14.4", "15.1"] },
    { "id": 5, "tasks": ["12.2", "15.2", "16.1", "17.1"] },
    { "id": 6, "tasks": ["16.2"] }
  ]
}
```
