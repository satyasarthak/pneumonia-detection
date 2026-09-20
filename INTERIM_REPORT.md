# Pneumonia Detection from Chest X-rays — Interim Report

**Sections:** Data Overview · Exploratory Data Analysis · Data Preprocessing · Model Building

---

## 1. Business Context and Objective

Pneumonia is a leading cause of illness and death worldwide, and chest X-rays
are the primary diagnostic tool. Reliable interpretation depends on skilled
radiologists, who are scarce in many regions and affected by fatigue and heavy
caseloads. The objective of this project is an automated decision-support
system that classifies a chest X-ray into one of three findings — **Normal**,
**Lung Opacity** (the pattern associated with pneumonia), or **No Lung Opacity
/ Not Normal** (a genuine abnormality that can look like pneumonia). This
interim report covers the foundation: understanding the data, preparing it, and
establishing a baseline model.

---

## 2. Data Overview

- **Source:** RSNA Pneumonia Detection dataset. Images are stored as DICOM
  medical-imaging files; labels are provided in a CSV.
- **Per-patient labels:** The label file lists one row per bounding box, so a
  pneumonia-positive patient can appear on several rows. For a per-patient
  classification task we collapse to one label per patient.

| Step | Records |
| --- | --- |
| Raw label rows | 30,227 |
| Unique patients (after de-duplication) | 26,684 |
| After restricting to the 3 valid classes | 26,684 |
| Patients with a matching image | 26,684 |
| Patients missing an image | 0 |

**Observation.** The dataset resolves to a clean set of 26,684 unique patients,
each with exactly one of the three labels and an available X-ray. This is a
sound basis for a fair three-class classification problem.

---

## 3. Exploratory Data Analysis

### 3.1 Class distribution

| Class | Patients | Share |
| --- | --- | --- |
| No Lung Opacity / Not Normal | 11,821 | 44.30% |
| Normal | 8,851 | 33.17% |
| Lung Opacity (pneumonia) | 6,012 | 22.53% |

*Figure: `outputs/class_distribution.png`*

**Observation — class imbalance.** The largest class has about **1.97×** the
samples of the smallest. The most under-represented class is "Lung Opacity"
(true pneumonia) — the very class we most care about. Left unaddressed, a model
would tend to under-predict pneumonia, so we apply class weighting during
training.

*Figure (pie view): `outputs/class_share_pie.png`*

### 3.2 Sample images per class

*Figure: `outputs/sample_images_per_class.png`*

**Observation.** X-rays vary in brightness, and the visual difference between
true pneumonia and the "look-alike" abnormal class is subtle. This subtlety is
the core modeling challenge and the main expected source of misclassification.

### 3.3 Patient demographics (age and sex)

Sampled from DICOM headers (~300 images per class).

| Class | Mean age | % Female | % Male |
| --- | --- | --- | --- |
| Normal | 44.6 | 47% | 53% |
| Lung Opacity (pneumonia) | 45.9 | 40% | 60% |
| No Lung Opacity / Not Normal | 49.7 | 46% | 54% |

*Figures: `outputs/eda_age.png`, `outputs/eda_sex_view.png`*

**Observation.** Age is broadly similar across classes (mid-40s), so age is not
a strong differentiator. Sex is fairly balanced, with pneumonia cases skewing
slightly male.

### 3.4 View position — an important confound

The X-ray **view position** differs sharply by class:

| Class | % PA | % AP |
| --- | --- | --- |
| Normal | 85% | 15% |
| No Lung Opacity / Not Normal | 43% | 57% |
| Lung Opacity (pneumonia) | 20% | 80% |

*Figure: `outputs/eda_sex_view.png`*

**Observation — this matters.** AP (anterior–posterior) films are typically
taken of sicker, bed-bound patients, whereas standard PA films are for
ambulatory patients. Pneumonia cases here are **80% AP** while Normal cases are
**85% PA**. This means some of the "signal" separating the classes is
acquisition-related, not purely pathological — a model could partly learn
"this looks like an AP film" as a proxy for pneumonia. This is a known bias in
chest-X-ray datasets and should be flagged for any real deployment.

### 3.5 Pixel intensity and image dimensions

*Figures: `outputs/eda_intensity.png`, `outputs/eda_dimensions.png`*

- **Mean pixel intensity** is nearly identical across classes (~121–126 on a
  0–255 scale), so overall brightness cannot separate them — the model must
  learn spatial patterns, not global exposure.
- **Image dimensions** are uniform at **1024×1024** across the sample, so a
  single resize target works for essentially all images.

### 3.6 Average (mean) X-ray per class

*Figure: `outputs/eda_mean_images.png`*

**Observation.** Averaging ~150 images per class shows subtle differences in
where lung-field intensity concentrates. The pneumonia mean image is hazier in
the lower lung fields, versus a crisper Normal mean image — visual confirmation
that a learnable spatial signal exists, even if it is fine-grained.

### 3.7 Key EDA takeaways

1. Three clean classes, one label per patient, no missing images.
2. Moderate class imbalance (~2:1) that must be handled in training.
3. The pneumonia-mimicking class is the largest, making the
   pneumonia-vs-look-alike boundary the key difficulty.
4. **View position is confounded with the label** (pneumonia mostly AP, Normal
   mostly PA) — a bias to monitor.
5. Classes are not separable by brightness or age alone; the model must learn
   spatial lung patterns.
6. Images are a uniform 1024×1024, simplifying preprocessing.

---

## 4. Data Preprocessing

The preprocessing pipeline turns a raw DICOM into a consistent, model-ready
tensor:

1. **Decode** the DICOM into a pixel array.
2. **Resize** to a fixed resolution (128×128 for the baseline).
3. **Normalize** pixel values to the 0–1 range (preserving relative
   intensities) so all images are on the same scale.
4. **Convert channels** — grayscale (1 channel) for the from-scratch CNN.
   Chest X-rays are inherently grayscale, so this loses no diagnostic detail.

*Figure (before/after): `outputs/preprocess_before_after.png`*

Images are streamed in small batches by a custom data generator, so the full
~26k-image dataset never has to fit in memory at once.

### 4.1 Train / validation / test split

The data is split **70% / 15% / 15%**, stratified so each class keeps its
proportion in every subset.

| Subset | Count | Normal | Lung Opacity | No Lung Opacity / Not Normal |
| --- | --- | --- | --- | --- |
| Train | 18,678 | 33.17% | 22.53% | 44.30% |
| Validation | 4,003 | 33.18% | 22.53% | 44.29% |
| Test | 4,003 | 33.15% | 22.53% | 44.32% |

**Observation.** Class proportions are nearly identical across subsets
(within 0.03%), confirming a fair, representative split.

---

## 5. Model Building — CNN from Scratch

### 5.1 Architecture

A convolutional neural network built from scratch as the baseline:

```
Input (128×128×1)
  → [Conv2D(32) → BatchNorm → ReLU → MaxPool]
  → [Conv2D(64) → BatchNorm → ReLU → MaxPool]
  → [Conv2D(128) → BatchNorm → ReLU → MaxPool]
  → GlobalAveragePooling → Dropout(0.5)
  → Dense(128, ReLU) → Dense(3, softmax)
```

The model compiles with categorical cross-entropy and is trained with
**class weights** derived from training-set frequencies to counter the
imbalance.

### 5.2 Training and performance

The baseline was trained on a balanced sample for a bounded number of epochs
(a full-dataset, high-resolution run requires a GPU). On the held-out test set:

| Metric | Value |
| --- | --- |
| Accuracy | 0.33 |
| Macro-F1 | 0.23 |

*Figures: `outputs/baseline_training_history.png`, `outputs/confusion_baseline_cnn.png`*

**Interpretation.** The from-scratch CNN learns only weakly under this limited
budget and tends toward the majority prediction, keeping macro-F1 low. This is
expected for a model trained from scratch on a small sample and short schedule.
It should be read as a **reference baseline**, not the ceiling of the approach.

---

## 6. Interim Takeaways

- The dataset is clean, three-class, and moderately imbalanced (~2:1), with the
  pneumonia class under-represented.
- Preprocessing standardizes image size, scale, and channels; the stratified
  split gives a fair evaluation set.
- A from-scratch CNN establishes a baseline but is not yet clinically useful.
- **Next (final submission):** transfer learning with pretrained backbones is
  expected to substantially outperform the baseline — this is the planned path
  to a usable model.

---

### Appendix — Reproducibility

All results are produced by a tested Python package (`src/`) and reproduced in
the accompanying notebook (`interim_submission.ipynb` / `.html`). Figures are
regenerated by `run_eda.py` and `run_train_baseline.py`. Numbers are reported to
two decimal places.
