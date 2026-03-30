FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ حمّل الـ MobileNet weights وقت البناء عشان ما يحتاج يحملها كل startup
COPY download_weights.py .
RUN python download_weights.py

COPY . .
EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
