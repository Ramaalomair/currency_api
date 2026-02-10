FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and compilation
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

# Uninstall any existing opencv-python and install headless version
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py currency_recognition.py ./

# Copy model directory
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
