FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py currency_recognition.py ./

# Copy model directory
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
