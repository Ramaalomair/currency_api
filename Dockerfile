FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py currency_recognition.py ./

# Create models directory (gdown will download model here)
RUN mkdir -p models/currency

# Expose port
EXPOSE 8080

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
