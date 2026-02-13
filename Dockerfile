FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install packages - opencv-headless MUST be installed before inference-sdk
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.104.1 \
        uvicorn[standard]==0.24.0 \
        python-multipart==0.0.6 \
        pillow==10.1.0 \
        numpy==1.26.2 \
        scikit-learn==1.6.1 \
        joblib==1.3.2 \
        torch==2.2.0 \
        torchvision==0.17.0 \
        opencv-python-headless==4.8.1.78 && \
    pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir inference-sdk

# Verify opencv-headless is installed (not opencv-python)
RUN python -c "import cv2; print('OpenCV version:', cv2.__version__)" && \
    pip list | grep opencv

# Copy application files
COPY main.py currency_recognition.py ./

# Copy model directory
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
