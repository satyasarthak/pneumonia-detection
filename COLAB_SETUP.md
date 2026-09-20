# Training on Google Colab (Free GPU) — Step by Step

This guide walks you through running `colab_train_gpu.ipynb` on Google Colab to
train the full transfer-learning suite on the complete dataset with a free GPU,
then bringing the strong model back into your project.

---

## Part 1 — Upload the data to Google Drive (do this first)

The DICOM archive is large (~3.7 GB) and is **not** in the GitHub repo, so Colab
needs to read it from your Drive.

1. Go to https://drive.google.com (sign in with the same Google account you use
   for Colab).
2. Create a new folder named exactly **`pneumonia_data`**.
3. Upload these two files from your project folder
   (`C:\Users\Sathya\Desktop\AI Project\CapstoneProject`) into that folder:
   - `stage_2_detailed_class_info.csv`
   - `stage_2_train_images.zip`
4. Wait for the upload to finish (the zip takes a while — start it early). You
   only ever do this once; it stays in Drive.

---

## Part 2 — Open the notebook in Colab

1. Go to https://colab.research.google.com
2. **File → Open notebook → GitHub** tab.
3. Enter your repo: `satyasarthak/pneumonia-detection`
4. Open **`colab_train_gpu.ipynb`**.

(Alternatively: open Colab, File → Upload notebook, and upload the
`colab_train_gpu.ipynb` from your local folder.)

---

## Part 3 — Turn on the GPU

1. **Runtime → Change runtime type**
2. Hardware accelerator → **GPU** (T4 is fine) → **Save**.
3. Run **cell 1** — it prints the TensorFlow version and confirms a GPU is
   detected. If it asserts "No GPU", redo this step.

---

## Part 4 — Run the notebook

Run cells top to bottom (**Runtime → Run all**, or Shift+Enter per cell):

1. **Get the code** — clones your repo and installs `pydicom`.
2. **Get the data** — run the **Option A (Google Drive)** cell. Approve the
   Drive-mount popup (pick your account, allow access). It copies the two files
   into the Colab workspace. Skip the Option B cell.
3. **Prepare the dataset** — loads labels, resolves images, splits.
4. **Training cells** — trains MobileNetV2, ResNet50, the deeper-head variant,
   and the fine-tuned model. On a GPU each takes only a few minutes. Let them
   all run.
5. **Compare + select** — prints the comparison table and the chosen best model.
6. **Serialize + inference** — saves `best_model.keras`, verifies the reload,
   and prints sample predictions.
7. **Save results** — copies `best_model.keras` and the comparison CSV back to
   your Drive folder (or downloads them if Drive isn't mounted).

**Tip:** if the session disconnects, just Run all again — it's idempotent.

---

## Part 5 — Bring the strong model back into your project

1. Download **both** files from your `pneumonia_data` Drive folder (or the Colab
   download prompts):
   - `best_model.keras`
   - `best_model_geometry.txt`
2. Copy them into your local project's `models/` folder, replacing the existing
   files:
   `C:\Users\Sathya\Desktop\AI Project\CapstoneProject\models\`
3. **Geometry is automatic** — the app reads `best_model_geometry.txt` and
   configures its input size on its own (the Colab model is RGB 224×224). No
   manual editing of `app.py` or the `Dockerfile` is needed.
4. Commit and push:
   ```
   git add models/best_model.keras models/best_model_geometry.txt
   git commit -m "Use full-data GPU-trained best model"
   git push
   ```

---

## Part 6 — Fill in the report

1. Open `outputs/model_comparison_final.csv` (from Drive) — this is your final
   comparison table.
2. In `FINAL_REPORT.md`, replace every `[[..]]` placeholder in Section 6 with
   these numbers, name the best model, and write the selection rationale.
3. Add the per-class metrics for the best model (printed in the notebook).

---

## Troubleshooting

- **"No GPU"** → Runtime → Change runtime type → GPU.
- **"Folder not found: .../pneumonia_data"** → the Drive folder name must match
  exactly, and the two files must be inside it.
- **Session timed out mid-training** → Colab free sessions can disconnect;
  rerun. To reduce risk, you can lower `EPOCHS` in the training cell.
- **Out of memory** → lower `BATCH` from 32 to 16 in the training-config cell.
