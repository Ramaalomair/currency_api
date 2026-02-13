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

# Install packages in specific order to prevent opencv conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.104.1 \
        uvicorn[standard]==0.24.0 \
        python-multipart==0.0.6 \
        pillow==10.1.0 \
        numpy==1.26.2 \
        scikit-learn==1.6.1 \
        joblib==1.3.2 && \
    pip install --no-cache-dir \
        torch==2.2.0 \
        torchvision==0.17.0 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir --no-deps inference-sdk && \
    pip install --no-cache-dir \
        requests \
        aiohttp \
        orjson \
        urllib3 \
        certifi \
        charset-normalizer \
        idna \
        attrs \
        multidict \
        yarl \
        aiosignal \
        frozenlist \
        async-timeout

# Copy application files
COPY main.py currency_recognition.py ./

# Copy model directory
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
