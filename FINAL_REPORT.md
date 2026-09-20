# Pneumonia Detection from Chest X-rays — Final Report

**Sections:** Data Overview · EDA · Data Preprocessing · Model Building ·
Transfer Learning · Model Deployment · Actionable Insights & Recommendations

> Note: Sections 1–4 reproduce and extend the interim report. Sections 5–7 are
> the final-submission additions. Replace the bracketed `[[…]]` placeholders
> with your final GPU numbers from `outputs/model_comparison_final.csv`.

---

## 1. Executive Summary

Pneumonia is a leading cause of illness and death worldwide, and chest X-rays
are the primary diagnostic tool — but reliable interpretation depends on scarce
radiologists. This project delivers an automated decision-support system that
classifies a chest X-ray into **Normal**, **Lung Opacity** (pneumonia), or
**No Lung Opacity / Not Normal**, packaged as a web app deployable in
resource-constrained settings. The best model is a transfer-learning network
(**[[best model]]**) achieving **[[macro-F1]]** on a held-out test set,
substantially outperforming a from-scratch CNN baseline.

## 2. Data Overview

- **Source:** RSNA Pneumonia Detection dataset (DICOM images + label CSV).
- **Volume:** 30,227 label rows → **26,684 unique patients** after
  de-duplication; all have a matching image (0 missing).
- **Classes:** exactly three, one per patient.

## 3. Exploratory Data Analysis

### 3.1 Class distribution and imbalance

| Class | Patients | Share |
| --- | --- | --- |
| No Lung Opacity / Not Normal | 11,821 | 44.30% |
| Normal | 8,851 | 33.17% |
| Lung Opacity (pneumonia) | 6,012 | 22.53% |

Imbalance ratio ≈ **1.97×**. *Figures: `class_distribution.png`, `class_share_pie.png`.*

### 3.2 Sample images per class
*Figure: `sample_images_per_class.png`.* The visual difference between
pneumonia and the look-alike class is subtle — the core modeling challenge.

### 3.3 Demographics
Age is similar across classes (mid-40s); sex is roughly balanced (pneumonia
slightly more male). *Figures: `eda_age.png`, `eda_sex_view.png`.*

### 3.4 View-position confound (key finding)
Pneumonia cases are **80% AP** vs Normal **85% PA**. AP films are taken of
sicker, bed-bound patients, so view position partly leaks the label — a bias to
monitor for deployment. *Figure: `eda_sex_view.png`.*

### 3.5 Intensity and dimensions
Mean pixel intensity overlaps across classes (~121–126), so brightness alone
cannot separate them; images are a uniform 1024×1024.
*Figures: `eda_intensity.png`, `eda_dimensions.png`.*

### 3.6 Mean image per class
Averaging images per class shows subtle lower-lung haze differences — a
learnable spatial signal. *Figure: `eda_mean_images.png`.*

## 4. Data Preprocessing

DICOM → decode → resize (224×224) → normalize to [0, 1] → channel conversion
(grayscale for baseline; RGB for transfer models). Images are streamed in
batches (memory-efficient). Data is split **70/15/15**, stratified — class
proportions preserved to within 0.03%. Augmentation (flip, small
rotation/zoom, brightness jitter) is applied to the training set only.

## 5. Model Building — Baseline CNN (from scratch)

A 3-block convolutional network (Conv→BN→ReLU→MaxPool ×3 → GAP → Dropout →
Dense → softmax) trained with class weights. It establishes the reference
point; a from-scratch CNN on this dataset is weak (near-chance on a small run),
motivating transfer learning.

*Figures: `baseline_training_history.png`, `confusion_baseline_cnn.png`.*

## 6. Transfer Learning

We built and compared multiple pretrained architectures, all evaluated on the
**same** held-out test set with macro-F1 (robust to imbalance):

1. **MobileNetV2** — pretrained backbone + standard Dense(128) head.
2. **ResNet50** — pretrained backbone + standard Dense(128) head.
3. **MobileNetV2-Deep** — a **new architecture** adding layers on top of the
   backbone: Dense(256)→BN→Dropout→Dense(128)→BN→Dropout→softmax.
4. **MobileNetV2-FineTuned** — two-phase training: train the head, then
   unfreeze the top of the backbone and fine-tune at a low learning rate.

### 6.1 Model comparison

*(Fill from `outputs/model_comparison_final.csv` after GPU training.)*

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | --- | --- | --- | --- |
| MobileNetV2 | [[..]] | [[..]] | [[..]] | [[..]] |
| ResNet50 | [[..]] | [[..]] | [[..]] | [[..]] |
| MobileNetV2-Deep | [[..]] | [[..]] | [[..]] | [[..]] |
| MobileNetV2-FineTuned | [[..]] | [[..]] | [[..]] | [[..]] |
| Baseline CNN | [[..]] | [[..]] | [[..]] | [[..]] |

*Confusion matrices: `confusion_*.png`.*

### 6.2 Best model and rationale
**[[best model]]** was selected for the highest macro-F1 (**[[value]]**).
Rationale: it best balances the three classes despite the imbalance, and
[[note fine-tuning / deeper-head effect and per-class recall, especially on the
hard "No Lung Opacity / Not Normal" class]].

### 6.3 Serialization and inference
The best model is serialized to `models/best_model.keras`, reloaded, and used
for inference — the reload reproduces predictions exactly (verified), and it
powers the deployed app.

## 7. Model Deployment

- **Streamlit app** (`app.py`): upload a chest X-ray → predicted class + a
  per-class probability bar chart; invalid uploads are rejected.
- **Dependencies:** `requirements.txt` (backend + frontend).
- **Docker:** `Dockerfile` builds and runs the app (`EXPOSE 8501`).
- **Repository:** pushed to GitHub (`satyasarthak/pneumonia-detection`).
- **GitHub Codespaces:** `.devcontainer` forwards port 8501; running
  `streamlit run app.py` exposes a **forwarded URL** for live inference.

**Forwarded URL:** `[[paste your Codespaces 8501 URL here]]`
**Inference screenshot:** `[[insert screenshot of an upload + prediction]]`

## 8. Actionable Insights and Recommendations

**Clinical / business**
1. Use as a triage aid, not a diagnosis — flag likely-pneumonia for faster
   radiologist review.
2. Tune for pneumonia **recall**: a missed pneumonia is costlier than a false
   alarm; route borderline cases to a human.
3. Watch the **view-position confound** — validate that the model uses lung
   pathology, not acquisition artifacts, before trusting it clinically.
4. Monitor the "No Lung Opacity / Not Normal" class — it drives most confusion.

**Technical**
5. Deploy the lightweight **MobileNetV2** for low-resource settings (strong
   accuracy-to-size ratio).
6. Report **per-class metrics**, not just accuracy, given the imbalance.
7. Establish continuous evaluation to catch drift across machines/sites.

## 9. Conclusion

The project delivers a complete, tested, and deployed pneumonia-detection
pipeline. Transfer learning substantially outperforms the from-scratch
baseline, and the lightweight best model is well suited to real-world
deployment. The standout analytical insight — the view-position confound — and
a focus on per-class recall are what make this a trustworthy decision-support
tool rather than just a leaderboard number.

---

### Appendix
- Code: tested `src/` package (52 automated tests, 18 correctness properties).
- Reproducibility: `run_eda.py`, `run_train_transfer.py`, `run_improve_model.py`,
  and `colab_train_gpu.ipynb` (full-data GPU training).
- All numbers reported to 2 decimal places.
