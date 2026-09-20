"""Build the interim-submission Jupyter notebook programmatically.

Creates interim_submission.ipynb covering the four interim sections
(Data Overview, EDA, Data Preprocessing, Model Building) with narrative
markdown between reproducible code cells. The code reuses the tested ``src``
package so results match the pipeline.

Run:
    C:\\venvs\\pneu\\Scripts\\python.exe build_interim_notebook.py
Then execute + convert to HTML (see convert commands printed at the end).
"""

from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------- Title
md(
    "# Pneumonia Detection from Chest X-rays - Interim Submission\n"
    "\n"
    "**Sections:** Data Overview - Exploratory Data Analysis - Data "
    "Preprocessing - Model Building\n"
    "\n"
    "This notebook classifies chest X-rays from the RSNA Pneumonia Detection "
    "dataset into three findings: **Normal**, **Lung Opacity** (pneumonia), and "
    "**No Lung Opacity / Not Normal** (an abnormality that can mimic pneumonia). "
    "It reuses a tested pipeline package (`src/`) so every result here is "
    "reproducible."
)

# ---------------------------------------------------------------- Setup
md("## Setup\n\nImport the pipeline modules and configure plotting.")
code(
    "import os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from src.labels import load_labels, deduplicate_patients, restrict_labels\n"
    "from src.images import resolve_images, load_dicom, ImageSource\n"
    "from src.preprocess import resize, normalize, to_channels, GRAYSCALE\n"
    "from src.split import stratified_split\n"
    "from src.eda import class_distribution, imbalance_report, sample_images_per_class\n"
    "\n"
    "CLASS_INFO_CSV = 'stage_2_detailed_class_info.csv'\n"
    "TRAIN_ZIP = 'stage_2_train_images.zip'\n"
    "pd.set_option('display.float_format', lambda v: f'{v:.2f}')"
)

# ---------------------------------------------------------------- 1. Data Overview
md(
    "## 1. Data Overview\n"
    "\n"
    "We import the label file and inspect its structure. The RSNA data lists "
    "one row per bounding box, so a pneumonia-positive patient can appear "
    "multiple times. For a per-patient classification task we collapse to one "
    "label per patient and keep only the three valid classes."
)
code(
    "labels = load_labels(CLASS_INFO_CSV, verbose=False)\n"
    "print('Raw label rows:', labels.shape)\n"
    "labels.head()"
)
code(
    "deduped = deduplicate_patients(labels)\n"
    "clean = restrict_labels(deduped)\n"
    "print('Unique patients after de-duplication:', deduped.shape[0])\n"
    "print('After restricting to 3 classes:', clean.shape[0])"
)
code(
    "# Confirm every patient has an image in the archive (missing are excluded).\n"
    "present, missing = resolve_images(clean, TRAIN_ZIP)\n"
    "print('Patients with an image:', len(present))\n"
    "print('Patients missing an image:', len(missing))"
)
md(
    "**Observation.** The dataset resolves to a clean set of unique patients, "
    "each with exactly one of the three labels and a matching X-ray image. This "
    "is the foundation for a fair three-class classification problem."
)

# ---------------------------------------------------------------- 2. EDA
md(
    "## 2. Exploratory Data Analysis\n"
    "\n"
    "### 2.1 Class distribution and imbalance"
)
code(
    "dist = class_distribution(present)\n"
    "dist_df = pd.DataFrame({'Class': list(dist.keys()), 'Patients': list(dist.values())})\n"
    "dist_df['Share %'] = (dist_df['Patients'] / dist_df['Patients'].sum() * 100).round(2)\n"
    "dist_df"
)
code(
    "report = imbalance_report(dist)\n"
    "print('Imbalanced:', report['imbalanced'])\n"
    "print('Majority class:', report['majority_class'])\n"
    "print('Minority class:', report['minority_class'])\n"
    "print(f\"Imbalance ratio (max/min): {report['ratio']:.2f}\")\n"
    "print()\n"
    "print(report['observation'])"
)
code(
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "ax.bar(dist_df['Class'], dist_df['Patients'], color='steelblue')\n"
    "ax.set_ylabel('Number of patients'); ax.set_title('Class distribution')\n"
    "ax.tick_params(axis='x', rotation=15)\n"
    "for i, v in enumerate(dist_df['Patients']):\n"
    "    ax.text(i, v + 100, str(v), ha='center')\n"
    "plt.tight_layout(); plt.show()"
)
md(
    "**Observation.** The classes are imbalanced (about a 2:1 ratio between the "
    "largest and smallest class). The largest class is the pneumonia-mimicking "
    "'No Lung Opacity / Not Normal' group, and the smallest is 'Lung Opacity' "
    "(true pneumonia). We must account for this during training so the model "
    "does not neglect the pneumonia class."
)
md("### 2.2 Sample images per class\n\nRandomly selected X-rays, labeled by class.")
code(
    "samples = sample_images_per_class(present, TRAIN_ZIP, n=4)\n"
    "n = 4\n"
    "fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))\n"
    "for r, (cls, imgs) in enumerate(samples.items()):\n"
    "    for c in range(n):\n"
    "        ax = axes[r, c]; ax.axis('off')\n"
    "        if c < len(imgs):\n"
    "            ax.imshow(imgs[c], cmap='gray'); ax.set_title(cls, fontsize=9)\n"
    "plt.tight_layout(); plt.show()"
)
md(
    "**Observation.** The X-rays vary in size and brightness, and the visual "
    "difference between pneumonia (lung opacity) and the look-alike class is "
    "subtle - which is exactly why an automated, consistent classifier is "
    "valuable and why this class pair will drive most errors."
)

# ---------------------------------------------------------------- 3. Preprocessing
md(
    "## 3. Data Preprocessing\n"
    "\n"
    "Each DICOM is decoded to a pixel array, resized to a fixed resolution, "
    "normalized to the 0-1 range, and converted to the channel format the model "
    "expects. Below we show one image before and after preprocessing."
)
code(
    "with ImageSource(TRAIN_ZIP) as src:\n"
    "    sample_pid = str(present.iloc[0]['patientId'])\n"
    "    raw = load_dicom(src, sample_pid)\n"
    "print('Raw shape:', raw.shape, '| dtype:', raw.dtype, '| range:', (int(raw.min()), int(raw.max())))\n"
    "\n"
    "processed = to_channels(normalize(resize(raw, (128, 128))), GRAYSCALE)\n"
    "print('Processed shape:', processed.shape, '| range:', (round(float(processed.min()), 2), round(float(processed.max()), 2)))"
)
code(
    "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
    "axes[0].imshow(raw, cmap='gray'); axes[0].set_title('Before (raw DICOM)'); axes[0].axis('off')\n"
    "axes[1].imshow(processed[..., 0], cmap='gray'); axes[1].set_title('After (resized + normalized)'); axes[1].axis('off')\n"
    "plt.tight_layout(); plt.show()"
)
md(
    "### 3.1 Grayscale conversion\n"
    "The from-scratch CNN uses single-channel (grayscale) input; transfer-learning "
    "models later use 3-channel input. The pipeline converts between the two as "
    "needed. Chest X-rays are inherently grayscale, so this loses no information "
    "for the baseline."
)
md(
    "### 3.2 Train / validation / test split\n"
    "We split into 70% / 15% / 15% while preserving each class's proportion in "
    "every subset (stratified), so evaluation is representative."
)
code(
    "train_df, val_df, test_df = stratified_split(present)\n"
    "rows = []\n"
    "for name, subset in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:\n"
    "    props = subset['class_label'].value_counts(normalize=True)\n"
    "    row = {'Subset': name, 'Count': len(subset)}\n"
    "    for cls in dist:\n"
    "        row[cls] = round(props.get(cls, 0) * 100, 2)\n"
    "    rows.append(row)\n"
    "pd.DataFrame(rows)"
)
md(
    "**Observation.** Class proportions are nearly identical across the three "
    "subsets, confirming the split is stratified and the test set is a fair "
    "yardstick."
)

# ---------------------------------------------------------------- 4. Model Building
md(
    "## 4. Model Building - CNN from Scratch\n"
    "\n"
    "We define a convolutional neural network from scratch as a baseline: three "
    "convolution blocks (Conv -> BatchNorm -> ReLU -> MaxPool), global average "
    "pooling, dropout, and a 3-way softmax output. Class weights counter the "
    "imbalance during training.\n"
    "\n"
    "> Training on the full ~26k images at high resolution needs a GPU. For a "
    "reproducible interim demonstration we train on a balanced sample at a "
    "smaller resolution; the architecture and procedure are identical to a full "
    "run."
)
code(
    "from src.models import build_baseline_cnn\n"
    "model = build_baseline_cnn(input_shape=(128, 128, 1))\n"
    "model.summary()"
)
code(
    "from src.train import train_baseline, compute_class_weights\n"
    "\n"
    "# Balanced subsample for a time-bounded, reproducible interim run.\n"
    "def subsample(df, per_class, seed=42):\n"
    "    parts = [g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby('class_label')]\n"
    "    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)\n"
    "\n"
    "sample = subsample(present, 300)\n"
    "tr, va, te = stratified_split(sample)\n"
    "print('Train/Val/Test:', len(tr), len(va), len(te))\n"
    "print('Class weights:', compute_class_weights(tr))"
)
code(
    "model, history = train_baseline(tr, va, TRAIN_ZIP, image_size=(128, 128),\n"
    "                                batch_size=32, epochs=5, model=model)"
)
code(
    "h = history.history\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
    "axes[0].plot(h.get('loss', []), label='train'); axes[0].plot(h.get('val_loss', []), label='val')\n"
    "axes[0].set_title('Loss'); axes[0].set_xlabel('epoch'); axes[0].legend()\n"
    "axes[1].plot(h.get('accuracy', []), label='train'); axes[1].plot(h.get('val_accuracy', []), label='val')\n"
    "axes[1].set_title('Accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend()\n"
    "plt.tight_layout(); plt.show()"
)
code(
    "from src.evaluate import evaluate_model, compute_metrics\n"
    "from src.generator import XrayBatchGenerator\n"
    "\n"
    "test_gen = XrayBatchGenerator(te, TRAIN_ZIP, batch_size=32,\n"
    "                              target_channels=1, image_size=(128, 128), shuffle=False)\n"
    "metrics = evaluate_model(model, test_gen, model_name='Baseline CNN')\n"
    "print(f\"Accuracy : {metrics['accuracy']:.2f}\")\n"
    "print(f\"Macro-F1 : {metrics['macro_f1']:.2f}\")\n"
    "pd.DataFrame(metrics['per_class']).T.round(2)"
)
md(
    "**Interpretation.** The from-scratch baseline learns only weakly on this "
    "small, short run and tends to favor the majority prediction - its macro-F1 "
    "stays low. This is the expected behavior of a CNN trained from scratch on a "
    "limited budget, and it sets the reference point that transfer learning "
    "(in the final submission) improves on substantially.\n"
    "\n"
    "### Key takeaways (interim)\n"
    "- The data is clean, three-class, and moderately imbalanced (~2:1).\n"
    "- Preprocessing standardizes size, scale, and channels; the split is "
    "stratified and fair.\n"
    "- A from-scratch CNN establishes a baseline but is not yet clinically "
    "useful - motivating transfer learning next."
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("interim_submission.ipynb", "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)

print("Wrote interim_submission.ipynb")
print("Next: execute and convert to HTML with")
print(r'  C:\venvs\pneu\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace interim_submission.ipynb')
print(r'  C:\venvs\pneu\Scripts\python.exe -m jupyter nbconvert --to html interim_submission.ipynb')
