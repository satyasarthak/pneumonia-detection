# Chest X-ray Pneumonia Detection

An end-to-end deep learning system that classifies chest X-ray radiographs
(RSNA Pneumonia Detection dataset) into three classes: **Normal**,
**Lung Opacity** (pneumonia), and **No Lung Opacity / Not Normal**. It covers
data ingestion, EDA, DICOM preprocessing, a baseline CNN, transfer learning,
model comparison/selection, and a deployable Streamlit app.

## Project layout

```
src/            Core package (data, models, training, evaluation, inference)
tests/          Property-based and example/smoke tests
app.py          Streamlit app (upload -> predicted class + probabilities)
run_eda.py            Data overview + EDA + preprocessing figures
run_train_baseline.py Train the from-scratch baseline CNN
run_evaluate.py       Evaluate + compare models, select best
run_registry_demo.py  Serialize best model, reload, run inference
Dockerfile      Container image for the Streamlit app
.devcontainer/  GitHub Codespaces config (forwards port 8501)
```

## Local setup

The project uses a Python 3.12 virtual environment (TensorFlow is installed
into a short path to avoid the Windows long-path limit):

```powershell
py -3.12 -m venv C:\venvs\pneu
C:\venvs\pneu\Scripts\python.exe -m pip install -r requirements.txt
```

Run the pipeline stages:

```powershell
C:\venvs\pneu\Scripts\python.exe run_eda.py
C:\venvs\pneu\Scripts\python.exe run_train_baseline.py --sample-per-class 0 --epochs 20 --image-size 224
C:\venvs\pneu\Scripts\python.exe run_evaluate.py
C:\venvs\pneu\Scripts\python.exe run_registry_demo.py
```

Run the tests:

```powershell
C:\venvs\pneu\Scripts\python.exe -m pytest tests/ -q
```

## Run the app locally

```powershell
C:\venvs\pneu\Scripts\streamlit.exe run app.py
```

Then open http://localhost:8501 and upload a chest X-ray (DICOM or image).

## Docker

```bash
docker build -t pneumonia-app .
docker run -p 8501:8501 pneumonia-app
```

For an RGB transfer model instead of the grayscale baseline, override the
input geometry:

```bash
docker run -p 8501:8501 -e TARGET_CHANNELS=3 -e IMAGE_HEIGHT=224 -e IMAGE_WIDTH=224 pneumonia-app
```

## Deploy on GitHub Codespaces

1. Push this repository to GitHub (the data `*.zip` files are git-ignored; keep
   them out of the repo and download separately). Ensure
   `models/best_model.keras` is committed so the app has a model to serve.
2. On GitHub, click **Code -> Codespaces -> Create codespace on main**.
3. The devcontainer installs dependencies automatically (`postCreateCommand`).
4. In the Codespace terminal, start the app:
   ```bash
   streamlit run app.py
   ```
5. Codespaces forwards port **8501** and shows a **forwarded URL**
   (Ports tab). Open it to use the app.
6. Upload a chest X-ray on the forwarded URL to run live inference and see the
   predicted class with per-class probabilities.

> Decision-support demonstration only - not a substitute for professional
> medical diagnosis.
