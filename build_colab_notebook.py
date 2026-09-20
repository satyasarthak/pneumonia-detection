"""Build a self-contained Google Colab GPU training notebook.

Produces colab_train_gpu.ipynb: a notebook that runs entirely on Colab's free
GPU, trains the full transfer-learning suite on the full RSNA dataset, compares
all models, fine-tunes the winner, and saves the best model + figures. This is
the recommended path to strong, defensible final-submission numbers.

Run locally to (re)generate the notebook:
    C:\\venvs\\pneu\\Scripts\\python.exe build_colab_notebook.py
"""

from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))


md(
    "# Pneumonia Detection - GPU Training (Google Colab)\n"
    "\n"
    "This notebook trains the full transfer-learning suite on the **complete** "
    "RSNA dataset using Colab's free GPU, then compares models, fine-tunes the "
    "winner, and saves the best model. It reuses the tested `src/` package from "
    "the project repository.\n"
    "\n"
    "**Before you start:** Runtime -> Change runtime type -> Hardware "
    "accelerator -> **GPU**."
)

md("## 1. Confirm GPU is available")
code(
    "import tensorflow as tf\n"
    "print('TensorFlow:', tf.__version__)\n"
    "gpus = tf.config.list_physical_devices('GPU')\n"
    "print('GPUs:', gpus)\n"
    "assert gpus, 'No GPU! Set Runtime -> Change runtime type -> GPU.'"
)

md(
    "## 2. Get the code\n"
    "Clone your project repository (it contains the tested `src/` package and "
    "the app). Replace the URL if your repo differs."
)
code(
    "!git clone https://github.com/satyasarthak/pneumonia-detection.git\n"
    "%cd pneumonia-detection\n"
    "!pip -q install pydicom"
)

md(
    "## 3. Get the data\n"
    "The DICOM archives are NOT in the repo (too large). Provide the RSNA data "
    "one of these ways, then set the paths below:\n"
    "\n"
    "- **Kaggle API** (fastest): upload your `kaggle.json`, then download the "
    "RSNA Pneumonia Detection Challenge data; or\n"
    "- **Google Drive**: upload `stage_2_train_images.zip` and "
    "`stage_2_detailed_class_info.csv` to Drive and mount it.\n"
    "\n"
    "Below is the Google Drive option (simplest)."
)
code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "# EDIT these to point at your uploaded files in Drive:\n"
    "import shutil, os\n"
    "DRIVE = '/content/drive/MyDrive/pneumonia_data'  # folder in your Drive\n"
    "shutil.copy(os.path.join(DRIVE, 'stage_2_detailed_class_info.csv'), '.')\n"
    "shutil.copy(os.path.join(DRIVE, 'stage_2_train_images.zip'), '.')\n"
    "print('Data copied:', os.path.exists('stage_2_train_images.zip'))"
)

md(
    "## 4. Prepare the dataset\n"
    "Load labels, de-duplicate patients, restrict to the three classes, resolve "
    "images, and split (stratified). Uses the FULL dataset."
)
code(
    "import pandas as pd\n"
    "from src.labels import load_labels, deduplicate_patients, restrict_labels\n"
    "from src.images import resolve_images\n"
    "from src.split import stratified_split\n"
    "\n"
    "clean = restrict_labels(deduplicate_patients(load_labels('stage_2_detailed_class_info.csv', verbose=False)))\n"
    "present, missing = resolve_images(clean, 'stage_2_train_images.zip')\n"
    "print('Patients:', len(present), '| missing images:', len(missing))\n"
    "\n"
    "train_df, val_df, test_df = stratified_split(present)\n"
    "print('Train/Val/Test:', len(train_df), len(val_df), len(test_df))"
)

md(
    "## 5. Train the transfer-learning suite (full data, GPU)\n"
    "We train two pretrained backbones with the standard head, plus a DEEPER "
    "custom-head architecture (added layers), all at 224x224 with augmentation "
    "and class weights. On a GPU each is fast."
)
code(
    "from src.train import train_transfer, train_transfer_finetune\n"
    "from src.generator import XrayBatchGenerator\n"
    "from src.evaluate import evaluate_model, compare_models, select_best, commentary\n"
    "from src.models import build_transfer_model_deep\n"
    "from src.train import make_generators, compute_class_weights, _default_callbacks\n"
    "\n"
    "SIZE = (224, 224)\n"
    "BATCH = 32\n"
    "EPOCHS = 15\n"
    "results, models = {}, {}"
)
code(
    "# --- MobileNetV2 (standard head, frozen base) ---\n"
    "m, _ = train_transfer('MobileNetV2', train_df, val_df, 'stage_2_train_images.zip',\n"
    "                      image_size=SIZE, batch_size=BATCH, epochs=EPOCHS)\n"
    "models['MobileNetV2'] = m\n"
    "tg = XrayBatchGenerator(test_df, 'stage_2_train_images.zip', batch_size=BATCH,\n"
    "                        target_channels=3, image_size=SIZE, shuffle=False)\n"
    "results['MobileNetV2'] = evaluate_model(m, tg, 'MobileNetV2')\n"
    "print(commentary(results['MobileNetV2']))"
)
code(
    "# --- ResNet50 (standard head, frozen base) ---\n"
    "m, _ = train_transfer('ResNet50', train_df, val_df, 'stage_2_train_images.zip',\n"
    "                      image_size=SIZE, batch_size=BATCH, epochs=EPOCHS)\n"
    "models['ResNet50'] = m\n"
    "tg = XrayBatchGenerator(test_df, 'stage_2_train_images.zip', batch_size=BATCH,\n"
    "                        target_channels=3, image_size=SIZE, shuffle=False)\n"
    "results['ResNet50'] = evaluate_model(m, tg, 'ResNet50')\n"
    "print(commentary(results['ResNet50']))"
)
code(
    "# --- MobileNetV2 with a DEEPER custom head (new architecture, added layers) ---\n"
    "deep = build_transfer_model_deep('MobileNetV2', input_shape=(*SIZE, 3))\n"
    "tr_gen, va_gen = make_generators(train_df, val_df, 'stage_2_train_images.zip',\n"
    "                                 target_channels=3, image_size=SIZE, batch_size=BATCH)\n"
    "deep.fit(tr_gen, validation_data=va_gen, epochs=EPOCHS,\n"
    "         class_weight=compute_class_weights(train_df),\n"
    "         callbacks=_default_callbacks(), verbose=2)\n"
    "models['MobileNetV2-Deep'] = deep\n"
    "tg = XrayBatchGenerator(test_df, 'stage_2_train_images.zip', batch_size=BATCH,\n"
    "                        target_channels=3, image_size=SIZE, shuffle=False)\n"
    "results['MobileNetV2-Deep'] = evaluate_model(deep, tg, 'MobileNetV2-Deep')\n"
    "print(commentary(results['MobileNetV2-Deep']))"
)
code(
    "# --- Fine-tuned MobileNetV2 (two-phase: head, then unfreeze top of backbone) ---\n"
    "ft, hist = train_transfer_finetune('MobileNetV2', train_df, val_df,\n"
    "                                   'stage_2_train_images.zip', image_size=SIZE,\n"
    "                                   batch_size=BATCH, head_epochs=8, finetune_epochs=12)\n"
    "models['MobileNetV2-FineTuned'] = ft\n"
    "tg = XrayBatchGenerator(test_df, 'stage_2_train_images.zip', batch_size=BATCH,\n"
    "                        target_channels=3, image_size=SIZE, shuffle=False)\n"
    "results['MobileNetV2-FineTuned'] = evaluate_model(ft, tg, 'MobileNetV2-FineTuned')\n"
    "print(commentary(results['MobileNetV2-FineTuned']))"
)

md("## 6. Compare all models and select the best")
code(
    "table = compare_models(results).sort_values('macro_f1', ascending=False)\n"
    "display(table.round(4))\n"
    "best_name, rationale = select_best(table, metric='macro_f1')\n"
    "print('BEST:', best_name)\n"
    "print(rationale)"
)
code(
    "# Per-class detail for the best model.\n"
    "import pandas as pd\n"
    "pd.DataFrame(results[best_name]['per_class']).T.round(4)"
)

md("## 7. Serialize the best model, reload, and run inference")
code(
    "from src.registry import save_model, load_model\n"
    "import numpy as np\n"
    "\n"
    "save_model(models[best_name], 'models/best_model.keras')\n"
    "reloaded = load_model('models/best_model.keras')\n"
    "\n"
    "tg = XrayBatchGenerator(test_df.head(6), 'stage_2_train_images.zip', batch_size=6,\n"
    "                        target_channels=3, image_size=SIZE, shuffle=False)\n"
    "x, y = tg[0]\n"
    "before = models[best_name].predict(x, verbose=0)\n"
    "after = reloaded.predict(x, verbose=0)\n"
    "print('Round-trip identical:', np.allclose(before, after, atol=1e-6))\n"
    "\n"
    "from src.constants import INDEX_TO_CLASS\n"
    "for i in range(x.shape[0]):\n"
    "    p = after[i]; idx = int(p.argmax())\n"
    "    print(f'true={INDEX_TO_CLASS[int(y[i].argmax())]:<28} pred={INDEX_TO_CLASS[idx]:<28} p={p[idx]:.2f}')"
)

md(
    "## 8. Save results back to Drive\n"
    "Download `models/best_model.keras` and commit it to your repo so the "
    "Streamlit app serves the strong model. Also save the comparison table for "
    "your report."
)
code(
    "import shutil\n"
    "table.to_csv('outputs/model_comparison_final.csv', index=False)\n"
    "shutil.copy('models/best_model.keras', DRIVE)\n"
    "shutil.copy('outputs/model_comparison_final.csv', DRIVE)\n"
    "print('Saved best_model.keras and comparison to your Drive folder.')\n"
    "print('Next: download best_model.keras, replace it in your repo, and push.')"
)

md(
    "---\n"
    "### After training\n"
    "1. Download `best_model.keras` (from Drive or the file browser).\n"
    "2. In your local repo, replace `models/best_model.keras` with this file "
    "and update the app geometry if needed (it is RGB 224x224).\n"
    "3. Commit and push, then redeploy the Codespace so the app serves the "
    "strong model.\n"
    "4. Copy the comparison table and per-class metrics into your final report."
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": []},
}
with open("colab_train_gpu.ipynb", "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print("Wrote colab_train_gpu.ipynb")
