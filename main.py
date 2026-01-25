import sys
print("=" * 60, file=sys.stderr)
print("🚀 MAIN.PY STARTING...", file=sys.stderr)
print("=" * 60, file=sys.stderr)
sys.stderr.flush()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import os
from pathlib import Path
import asyncio

# استيراد دوال التعرف على العملة
from currency_recognition import (
    currency_recognizer,
    initialize_currency_recognition,
    recognize_currency_from_bytes,
    get_currency_recognition_status
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Currency Recognition API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# متغير عام للتأكد من تحميل الموديل مرة واحدة
MODEL_LOADED = False

@app.on_event("startup")
async def startup_event():
    """تحميل الموديل عند بداية التطبيق مرة واحدة فقط"""
    global MODEL_LOADED
    logger.info("🔄 Starting model initialization...")
    
    try:
        # تحقق إذا الموديل محمّل
        status = get_currency_recognition_status()
        if not status.get('initialized', False):
            logger.info("📥 Downloading and loading model...")
            await asyncio.to_thread(initialize_currency_recognition)
        
        MODEL_LOADED = True
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        MODEL_LOADED = False

@app.get("/")
async def root():
    return {
        "status": "online",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """فحص صحة الـ API والموديل"""
    status = get_currency_recognition_status()
    return {
        "status": "healthy" if MODEL_LOADED else "initializing",
        "model_status": status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recognize")
async def recognize_currency(file: UploadFile = File(...)):
    """التعرف على العملة من الصورة"""
    
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading, please try again in a moment"
        )
    
    try:
        # قراءة الصورة
        contents = await file.read()
        
        # التعرف على العملة
        result = await asyncio.to_thread(
            recognize_currency_from_bytes,
            contents
        )
        
        return {
            "success": True,
            "currency": result.get('currency'),
            "confidence": result.get('confidence'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
