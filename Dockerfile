# Pneumonia Detection - Streamlit app container
# Builds a portable image that serves the chest X-ray classifier.
FROM python:3.11-slim

# System libs needed by opencv / pillow / pydicom image handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, source package, and the serialized model.
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/

# Streamlit serves on 8501.
EXPOSE 8501

# Model input geometry for the served model (MobileNetV2: RGB, 160x160).
# Override at run time for a different model, e.g. grayscale baseline:
#   -e TARGET_CHANNELS=1 -e IMAGE_HEIGHT=128 -e IMAGE_WIDTH=128
ENV MODEL_PATH=models/best_model.keras \
    TARGET_CHANNELS=3 \
    IMAGE_HEIGHT=160 \
    IMAGE_WIDTH=160

# Basic container healthcheck against Streamlit's health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
