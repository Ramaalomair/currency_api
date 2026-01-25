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
import gdown
from pathlib import Path

# استيراد دوال التعرف على العملة
from currency_recognition import (
    currency_recognizer,
    initialize_currency_recognition,
    recognize_currency_from_bytes,
    get_currency_recognition_status
)

# إعداد الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# طباعة PORT للتأكد
logger.info(f"🔌 PORT from environment: {os.environ.get('PORT', 'Not set, using default 8000')}")

# تحميل الموديل من Google Drive عند أول تشغيل
MODEL_DIR = Path("models/currency")
MODEL_PATH = MODEL_DIR / "SVM_(RBF).pkl"
MODEL_FILE_ID = "1NUlvBjgPkej4WdNFL0WJFY43yTPz1M4n"

def download_model_if_needed():
    """تحميل الموديل من Google Drive إذا لم يكن موجوداً"""
    if not MODEL_PATH.exists():
        logger.info("📥 Downloading model from Google Drive...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(id=MODEL_FILE_ID, output=str(MODEL_PATH), quiet=False)
            logger.info("✅ Model downloaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise
    else:
        logger.info("✅ Model already exists, skipping download")

# تحميل الموديل عند بدء التطبيق
download_model_if_needed()

# تحميل موديلات التعرف على العملة
logger.info("🚀 Initializing currency recognition models...")
if initialize_currency_recognition():
    logger.info("✅ Currency recognition initialized successfully!")
else:
    logger.error("❌ Failed to initialize currency recognition!")

# إنشاء التطبيق
app = FastAPI(
    title="Munir Currency Recognition API",
    description="API for recognizing Saudi Arabian currency denominations",
    version="1.0.0"
)

# متغير لتتبع حالة التحميل
startup_complete = False

@app.on_event("startup")
async def startup_event():
    """تشغيل عند بدء التطبيق"""
    global startup_complete
    logger.info("🎉 Application startup complete - Ready to serve!")
    startup_complete = True

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قائمة العملات المدعومة
SUPPORTED_CURRENCIES = ["5 SR", "10 SR", "20 SR", "50 SR", "100 SR", "200 SR", "500 SR"]

@app.get("/")
async def root():
    """معلومات عن الـ API"""
    return {
        "api": "Munir Currency Recognition API",
        "version": "1.0.0",
        "status": "running",
        "currency_recognition": "loaded" if currency_recognizer.is_loaded else "not_loaded",
        "supported_currencies": SUPPORTED_CURRENCIES,
        "languages": ["arabic", "english"],
        "startup_complete": startup_complete
    }

@app.get("/health")
async def health_check():
    """فحص صحة الـ API"""
    model_exists = MODEL_PATH.exists()
    models_loaded = currency_recognizer.is_loaded
    
    # لو لسه ما خلص startup - نرجع unhealthy لكن مع رسالة
    if not startup_complete:
        logger.info("⏳ Health check: Application is still starting up...")
        return {
            "status": "starting",
            "model_exists": model_exists,
            "models_loaded": models_loaded,
            "startup_complete": False,
            "message": "Application is still starting up, please wait...",
            "timestamp": datetime.now().isoformat()
        }
    
    # لو الموديلات مو محملة
    if not (model_exists and models_loaded):
        logger.warning("⚠️ Health check: Models not ready!")
        return {
            "status": "unhealthy",
            "model_exists": model_exists,
            "models_loaded": models_loaded,
            "startup_complete": startup_complete,
            "message": "Models not ready",
            "timestamp": datetime.now().isoformat()
        }
    
    # كل شيء تمام
    logger.info("✅ Health check: All systems operational")
    return {
        "status": "healthy",
        "model_exists": model_exists,
        "models_loaded": models_loaded,
        "startup_complete": startup_complete,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recognize_currency")
async def recognize_currency(file: UploadFile = File(...)):
    """
    التعرف على فئة العملة من الصورة
    
    Parameters:
    - file: صورة العملة
    
    Returns:
    - currency: فئة العملة (مثل: "100 SR")
    - confidence: نسبة الثقة
    - text: النص بالعربية والإنجليزية
    """
    try:
        logger.info(f"📸 Received currency recognition request: {file.filename}")
        
        # التحقق من تحميل الموديلات
        if not currency_recognizer.is_loaded:
            logger.error("❌ Currency recognition service not ready")
            raise HTTPException(
                status_code=503,
                detail="Currency recognition service not ready. Please wait for models to load."
            )
        
        # قراءة الصورة
        image_bytes = await file.read()
        
        # التعرف على العملة
        result = recognize_currency_from_bytes(image_bytes)
        
        if not result.get("success"):
            logger.error(f"❌ Recognition failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Recognition failed")
            )
        
        # تنسيق النتيجة للـ API
        response = {
            "denomination": result["currency"],
            "confidence": result["confidence_percent"],
            "text_arabic": result["text"]["arabic"],
            "text_english": result["text"]["english"],
            "currency_value": result["currency_value"],
            "currency_unit": result["currency_unit"],
            "all_probabilities": result.get("all_probabilities", {})
        }
        
        logger.info(f"✅ Recognition successful: {result['currency']} ({result['confidence_percent']:.2f}%)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/currency_status")
async def currency_status():
    """حالة خدمة التعرف على العملات"""
    status = get_currency_recognition_status()
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024) if MODEL_PATH.exists() else 0
    
    return {
        "service": "Currency Recognition",
        "status": "active" if status["loaded"] else "inactive",
        "model_loaded": status["loaded"],
        "model_exists": status["model_exists"],
        "model_size_mb": f"{model_size_mb:.2f}",
        "device": status["device"],
        "num_classes": status["num_classes"],
        "supported_currencies": status["classes"],
        "languages": status["languages"],
        "startup_complete": startup_complete,
        "last_check": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    # استخدام PORT من environment variable (Railway يحتاجه)
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
