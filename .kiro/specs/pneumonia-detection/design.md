# Design Document

## Overview

The Pneumonia Detection system is an end-to-end deep learning pipeline that classifies chest X-ray radiographs from the RSNA Pneumonia Detection dataset into three categories: **Normal**, **Lung Opacity (Pneumonia)**, and **No Lung Opacity / Not Normal**. It spans the full ML lifecycle: ingestion of DICOM imagery and CSV labels, exploratory data analysis, DICOM preprocessing, a baseline CNN trained from scratch, transfer learning with pretrained backbones, model comparison and selection, artifact serialization, and deployment as a Streamlit application containerized with Docker and served through GitHub Codespaces with live inference.

The design favors modular, testable components with clear input/output contracts so that the pure data-transformation and inference logic can be validated independently of heavyweight training and external deployment concerns.

### Technology Stack

| Concern | Library / Tool |
| --- | --- |
| DICOM decode | `pydicom` |
| Numeric / tabular | `numpy`, `pandas` |
| EDA visualization | `matplotlib`, `seaborn` |
| Splitting & metrics | `scikit-learn` |
| Modeling | `tensorflow` / `keras` (CNN + transfer learning) |
| Memory-efficient loading | `keras.utils.Sequence` generators / `tf.data` |
| Web application | `streamlit` |
| Packaging | `Docker`, `requirements.txt` |
| Hosting | GitHub repository + GitHub Codespaces |

Python is the implementation language; code examples in this document use Python.

## Architecture

The system is organized into five layers. The lower three layers (data, modeling, evaluation) contain the pure, testable logic. The upper two layers (serving, deployment) wrap that logic for user-facing delivery.

```
┌───────────────────────────────────────────────────────────────┐
│ Deployment Layer                                                │
│  Dockerfile · requirements.txt · GitHub repo · Codespaces URL   │
└───────────────────────────────────────────────────────────────┘
                              ▲
┌───────────────────────────────────────────────────────────────┐
│ Serving Layer (Streamlit_App)                                   │
│  upload → validate → preprocess → InferenceService → render     │
└───────────────────────────────────────────────────────────────┘
                              ▲
┌───────────────────────────────────────────────────────────────┐
│ Evaluation & Registry Layer                                     │
│  Model_Evaluator · ModelComparator · ModelSelector · Registry   │
└───────────────────────────────────────────────────────────────┘
                              ▲
┌───────────────────────────────────────────────────────────────┐
│ Modeling Layer                                                  │
│  Baseline_CNN · Transfer_Model(VGG16/ResNet50/MobileNetV2)      │
└───────────────────────────────────────────────────────────────┘
                              ▲
┌───────────────────────────────────────────────────────────────┐
│ Data Layer (Data_Pipeline + EDA_Module)                         │
│  LabelLoader · ImageResolver · Deduplicator · Preprocessor      │
│  Splitter(stratified) · BatchGenerator · EDA visualizations     │
└───────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

- **Data_Pipeline** — ingest labels, resolve images, de-duplicate patients, restrict labels, decode & preprocess DICOM, split, and generate batches.
- **EDA_Module** — sample and annotate images, compute class distribution, flag imbalance.
- **Modeling Layer** — construct the baseline CNN and the transfer-learning models with custom 3-node softmax heads.
- **Model_Evaluator / ModelComparator / ModelSelector** — compute multi-class metrics, build the comparison table, choose the best model with rationale.
- **Model_Registry** — serialize the selected model and reload it for inference.
- **InferenceService** — the shared prediction function used by both notebooks and the Streamlit app (validate → preprocess → predict → map to label + probabilities).
- **Streamlit_App / Deployment_Package** — user upload UI, dependency manifest, Docker image, Codespaces deployment.
- **Report_Module** — insights, clinical recommendations, and the structured business report.

## Components and Interfaces

### Data Layer

```python
# labels.py
def load_labels(class_info_csv: str) -> pd.DataFrame:
    """Parse Label_Source into columns [patientId, class_label]. (Req 1.1)"""

def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per patientId (first/majority label). (Req 1.3)"""

def restrict_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the three canonical Class_Label values. (Req 1.4)"""

# images.py
def resolve_images(df: pd.DataFrame, image_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """Split records into (present, missing_patient_ids). Missing excluded downstream. (Req 1.5)"""

def load_dicom(path: str) -> np.ndarray:
    """Decode a DICOM_Image into a 2D pixel array. (Req 3.1)"""

# preprocess.py
def to_channels(img: np.ndarray, target: str) -> np.ndarray:
    """Convert between grayscale (1ch) and RGB (3ch) for the target model. (Req 3.2)"""

def normalize(img: np.ndarray) -> np.ndarray:
    """Scale pixel values into [0, 1]. (Req 3.6)"""

def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to the model input resolution."""

# split.py
def stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42) -> tuple[df, df, df]:
    """Partition into train/val/test preserving class proportions. (Req 3.4, 3.5)"""

# generator.py
class XrayBatchGenerator(keras.utils.Sequence):
    """Loads DICOM images incrementally in batches. (Req 3.7)"""
    def __init__(self, records, batch_size, target_channels, image_size, augment=False): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]: ...
```

`XrayBatchGenerator` reads only the images referenced by the current batch's indices, satisfying the memory-efficiency constraint for the ~26k-image dataset. Optional on-the-fly augmentation is one lever for class imbalance (see Error Handling / Imbalance).

### EDA Module

```python
def sample_images_per_class(df, image_dir, n=5) -> dict[str, list[np.ndarray]]:
    """Randomly sample n images per Class_Label for display. (Req 2.1, 2.2)"""

def class_distribution(df) -> dict[str, int]:
    """Count of patientId records per Class_Label. (Req 2.3)"""

def imbalance_report(dist: dict[str, int], threshold: float = 1.5) -> dict:
    """Flag imbalance when max/min class ratio exceeds threshold. (Req 2.4)"""
```

### Modeling Layer

```python
# models.py
def build_baseline_cnn(input_shape=(224, 224, 1), num_classes=3) -> keras.Model:
    """CNN from scratch with a 3-node softmax output. (Req 4.1)"""

def build_transfer_model(backbone: str,           # "VGG16" | "ResNet50" | "MobileNetV2"
                         input_shape=(224, 224, 3),
                         num_classes=3,
                         freeze_base=True) -> keras.Model:
    """Pretrained backbone + custom 3-node softmax head. (Req 5.1, 5.2)"""
```

**Baseline_CNN architecture (from scratch):**

```
Input(224x224x1)
 → [Conv2D(32,3x3) → BN → ReLU → MaxPool] 
 → [Conv2D(64,3x3) → BN → ReLU → MaxPool]
 → [Conv2D(128,3x3) → BN → ReLU → MaxPool]
 → GlobalAveragePooling2D → Dropout(0.5)
 → Dense(128, ReLU) → Dense(3, softmax)
```

**Transfer_Model architecture (per backbone):**

```
Input(224x224x3)
 → <backbone>(weights="imagenet", include_top=False, trainable=freeze_base? False)
 → GlobalAveragePooling2D → Dropout(0.4)
 → Dense(128, ReLU) → Dense(3, softmax)
```

At least two of VGG16, ResNet50, MobileNetV2 are built (default: ResNet50 and MobileNetV2), plus the baseline, giving three models to compare. All models compile with categorical cross-entropy and support `class_weight` for imbalance.

### Evaluation, Comparison, Selection & Registry

```python
# evaluate.py
def evaluate_model(model, test_gen) -> dict:
    """Multi-class metrics: accuracy, macro precision/recall/F1,
       per-class report, confusion matrix. (Req 4.3, 5.3)"""

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model, identical metric columns for all. (Req 4.4, 5.4)"""

def select_best(comparison: pd.DataFrame, metric="macro_f1") -> tuple[str, str]:
    """Return (best_model_name, rationale) = argmax over metric. (Req 5.5)"""

# registry.py
def save_model(model, path: str) -> str:
    """Serialize to SavedModel dir or .h5. (Req 5.6)"""

def load_model(path: str) -> keras.Model:
    """Reload a serialized model. (Req 5.7)"""
```

### Serving Layer (shared inference)

```python
# inference.py
CLASSES = ["Normal", "Lung Opacity (Pneumonia)", "No Lung Opacity / Not Normal"]

def validate_upload(file_bytes: bytes, filename: str) -> None:
    """Raise UnsupportedImageError for non-DICOM/non-image inputs. (Req 6.3)"""

def preprocess_for_inference(img: np.ndarray, target_channels: int) -> np.ndarray:
    """Resize + channel-convert + normalize for the served model."""

def predict(model, img: np.ndarray) -> tuple[str, dict[str, float]]:
    """Return (predicted_label, {class: probability}). label = CLASSES[argmax].
       (Req 5.7, 6.1, 6.2)"""
```

### Streamlit App

```python
# app.py (structure)
model = load_model(MODEL_PATH)                    # cached with @st.cache_resource
uploaded = st.file_uploader(...)
if uploaded:
    try:
        validate_upload(uploaded.getvalue(), uploaded.name)   # Req 6.3
        img = load_dicom_or_image(uploaded)
        x = preprocess_for_inference(img, target_channels)
        label, probs = predict(model, x)                       # Req 6.1
        st.subheader(f"Prediction: {label}")
        st.bar_chart(probs)                                    # Req 6.2
    except UnsupportedImageError as e:
        st.error(str(e))                                       # Req 6.3
```

## Data Models

```python
@dataclass
class LabelRecord:
    patient_id: str
    class_label: str          # one of CLASSES (Req 1.4)

@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame        # disjoint, covering (Req 3.4)

@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray

@dataclass
class Prediction:
    label: str                # CLASSES[argmax(probabilities)]
    probabilities: dict[str, float]   # len 3, each in [0,1], sums to ~1 (Req 6.2)
```

- **Label_Source**: `stage_2_detailed_class_info.csv` → `patientId`, `class`.
- **Image archives**: `stage_2_train_images.zip`, `stage_2_test_images.zip` → `<patientId>.dcm`.
- Records with duplicate `patientId` are collapsed to one `LabelRecord` per patient.

## DICOM → Grayscale Preprocessing Flow

```
.dcm file
  │ pydicom.dcmread(path).pixel_array        (Req 3.1)
  ▼
raw 2D uint16 pixel array
  │ rescale to 8-bit, resize to (224,224)
  ▼
resized grayscale array
  │ normalize → [0,1]                         (Req 3.6)
  ▼
normalized array
  │ to_channels(target)                       (Req 3.2)
  │   - Baseline_CNN  → 1 channel (grayscale)
  │   - Transfer_Model → 3 channels (RGB via channel replication)
  ▼
model-ready tensor  ──►  BatchGenerator stacks into (batch, H, W, C)   (Req 3.7)
```

The EDA/preprocessing notebook renders a before/after pair for a sample image (Req 3.3), and sample grids annotated with class labels (Req 2.1, 2.2).

## Evaluation Approach (Multi-class Metrics)

All models are evaluated on the **same held-out test subset** using identical metrics so comparison is fair:

- **Overall accuracy**
- **Macro precision / recall / F1** (treat classes equally despite imbalance)
- **Per-class precision / recall / F1** via `classification_report`
- **Confusion matrix** (3×3), with each row summing to that class's support

Selection uses **macro-F1** by default (robust to class imbalance), with the rationale recorded as text comparing the selected model against the baseline and other transfer models. The confusion matrix and metric bar charts feed the business report visualizations (Req 8.3).

## Model Registry / Serialization

- The selected best model is serialized to a persistent artifact (`models/best_model` SavedModel directory, or `best_model.h5`).
- A round-trip check reloads the artifact and confirms identical predictions on a fixed input before deployment (Req 5.6, 5.7).
- The Streamlit app loads exactly this artifact, ensuring the deployed model is the evaluated best model.

## Deployment Design (Streamlit + Docker + Codespaces)

**Dependencies (`requirements.txt`)** — declares backend and frontend deps (Req 6.4):
```
streamlit
tensorflow
pydicom
numpy
pandas
pillow
scikit-learn
matplotlib
```

**Docker (`Dockerfile`)** — builds and runs the app in a container (Req 6.5):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Repository & Codespaces (Req 6.6–6.8):**
- Application code, model artifact, `requirements.txt`, and `Dockerfile` are pushed to a GitHub repository.
- A `.devcontainer` (or direct `streamlit run`) launches the app in a Codespace; port 8501 is forwarded, producing a public forwarded URL.
- Accessing that URL performs live inference: upload → validated → preprocessed → predicted label + per-class probabilities.

## Error Handling

- **Missing images (Req 1.5):** `resolve_images` partitions records into present/missing; missing patient IDs are logged and excluded from every downstream subset.
- **Invalid labels (Req 1.4):** rows whose label is outside the three canonical categories are dropped by `restrict_labels`.
- **Unsupported uploads (Req 6.3):** `validate_upload` raises `UnsupportedImageError`; the app catches it and shows an error without invoking the model.
- **Corrupt/undecodable DICOM (Req 3.1):** decode failures are caught, logged, and the record is skipped (treated like a missing image).

### Class Imbalance Handling

- **Class imbalance (Req 2.4):** detected in EDA; mitigated with `class_weight` computed from training-set frequencies and/or augmentation in the training generator.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Label parsing produces one labeled record per source row

*For any* valid Label_Source content, parsing SHALL yield exactly one record per source row, and every record SHALL carry a non-empty Patient_Id and a Class_Label.

**Validates: Requirements 1.1**

### Property 2: De-duplication yields unique patients

*For any* set of label records, after de-duplication each Patient_Id SHALL appear exactly once, the result SHALL contain no more records than the input, and every retained Patient_Id SHALL have existed in the input.

**Validates: Requirements 1.3**

### Property 3: Labels are restricted to three categories

*For any* input records (including records with arbitrary label strings), every Class_Label retained by the pipeline SHALL be one of exactly the three canonical categories.

**Validates: Requirements 1.4**

### Property 4: Missing images partition the records

*For any* set of records and image directory, the downstream (present) set and the missing set SHALL be disjoint, their union SHALL equal the input records, and every downstream record SHALL reference an existing image file.

**Validates: Requirements 1.5**

### Property 5: Class distribution aggregation is exact

*For any* set of records, the sum of the per-class counts SHALL equal the total number of records, and each per-class count SHALL equal the number of records bearing that Class_Label.

**Validates: Requirements 2.3**

### Property 6: Channel conversion produces the target format

*For any* pixel array, converting to the target representation SHALL produce grayscale output with 1 channel and RGB output with 3 channels, and converting grayscale→RGB→grayscale SHALL preserve the original spatial dimensions.

**Validates: Requirements 3.2**

### Property 7: Normalization bounds pixel values

*For any* pixel array with arbitrary numeric range, normalization SHALL map every value into the defined range [0, 1] and SHALL preserve the relative ordering of pixel intensities.

**Validates: Requirements 3.6**

### Property 8: Dataset partition is disjoint and covering

*For any* de-duplicated record set, the train, validation, and test subsets SHALL be pairwise disjoint and their union SHALL equal the input set, so no Patient_Id appears in more than one subset and none is lost.

**Validates: Requirements 3.4**

### Property 9: Stratified split preserves class proportions

*For any* input class distribution, the proportion of each Class_Label within each of the train, validation, and test subsets SHALL approximate the overall proportion of that Class_Label within a defined tolerance.

**Validates: Requirements 3.5**

### Property 10: Batched generator covers the dataset exactly once per epoch

*For any* dataset and batch size, iterating the generator for one epoch SHALL yield batches whose union equals the dataset with no record repeated, and every batch size SHALL be less than or equal to the configured batch size.

**Validates: Requirements 3.7**

### Property 11: Classification models produce a valid 3-class softmax output

*For any* classification model (baseline CNN or transfer model) and any input batch, the model output SHALL have shape (batch_size, 3) and each output row SHALL be a probability distribution whose entries are non-negative and sum to approximately 1.

**Validates: Requirements 4.1, 5.1, 5.2**

### Property 12: Multi-class metrics are valid

*For any* set of true labels and predicted labels over the three classes, the computed accuracy and macro precision/recall/F1 SHALL each lie in [0, 1], and every row of the confusion matrix SHALL sum to the number of test samples belonging to that class.

**Validates: Requirements 4.3, 5.3**

### Property 13: Model comparison table is complete and consistent

*For any* set of per-model evaluation results, the comparison table SHALL contain exactly one row per model and the identical set of metric columns for every model.

**Validates: Requirements 4.4, 5.4**

### Property 14: Best-model selection is the metric optimum

*For any* comparison table, the model selected as best SHALL have a selection-metric value greater than or equal to that of every other model in the table.

**Validates: Requirements 5.5**

### Property 15: Serialization round-trip preserves predictions

*For any* trained model and fixed input, saving the model to the registry and reloading it SHALL produce predictions equal to the original model's predictions within numerical tolerance.

**Validates: Requirements 5.6**

### Property 16: Predicted label maps to a valid category

*For any* model output probability vector, the predicted Class_Label SHALL equal the category at the argmax position and SHALL always be one of the three canonical categories.

**Validates: Requirements 5.7, 6.1**

### Property 17: Prediction probabilities are a valid distribution over three classes

*For any* valid uploaded image, the returned probabilities SHALL contain exactly one entry per Class_Label, each entry SHALL lie in [0, 1], the entries SHALL sum to approximately 1, and the displayed predicted label SHALL correspond to the maximum-probability class.

**Validates: Requirements 6.2**

### Property 18: Unsupported uploads are rejected

*For any* input that is not a supported chest X-ray image, upload validation SHALL raise an error and reject the input, and inference SHALL NOT be performed on it.

**Validates: Requirements 6.3**

## Testing Strategy

**Dual approach** — property-based tests validate universal behavior of the pure logic; example/integration/smoke tests cover display, wiring, infrastructure, and documentation criteria.

- **Property tests (min. 100 iterations each):** Properties 1–18 above. Each test is tagged `Feature: pneumonia-detection, Property {number}: {property_text}` and references its design property. Heavy model properties (11, 12, 15, 16) use tiny in-memory models and generated arrays/label vectors to stay fast and deterministic.
- **Example tests:** record-count reporting (1.2), sample display and annotation (2.1, 2.2), imbalance flag on a skewed fixture (2.4), before/after preprocessing plot (3.3), and the "≥2 distinct backbones built" check (5.1).
- **Integration tests (1–3 examples):** DICOM decode on real `.dcm` fixtures (3.1), a short `fit` run consuming train/val generators (4.2), and manual verification of the Codespaces forwarded URL with live inference (6.7, 6.8).
- **Smoke tests:** `requirements.txt` presence and required deps (6.4), Dockerfile build (6.5), repository contains app + artifacts (6.6).
- **Documentation criteria (7.1, 7.2, 8.1, 8.2, 8.3):** satisfied by the Report_Module deliverable and reviewed rather than unit-tested.

Property tests should focus on universal invariants across randomized inputs; unit/example tests should focus on specific scenarios, edge cases, and integration points, keeping the example set small since property tests provide broad input coverage.
