from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from currency_recognition import predict_currency_class
from datetime import datetime
import os
import gdown
from pathlib import Path

# إعداد الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# إنشاء التطبيق
app = FastAPI(
    title="Munir Currency Recognition API",
    description="API for recognizing Saudi Arabian currency denominations",
    version="1.0.0"
)

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
        "currency_recognition": "loaded",
        "supported_currencies": SUPPORTED_CURRENCIES,
        "languages": ["arabic", "english"]
    }

@app.get("/health")
async def health_check():
    """فحص صحة الـ API"""
    model_exists = MODEL_PATH.exists()
    return {
        "status": "healthy" if model_exists else "model_missing",
        "currency_recognition": model_exists,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recognize_currency")
async def recognize_currency(file: UploadFile = File(...)):
    """
    التعرف على فئة العملة من الصورة
    
    Parameters:
    - file: صورة العملة
    
    Returns:
    - denomination: فئة العملة (مثل: "100 SR")
    - confidence: نسبة الثقة
    - text_arabic: النص بالعربية
    - text_english: النص بالإنجليزية
    """
    try:
        logger.info(f"Received currency recognition request: {file.filename}")
        
        # قراءة الصورة
        image_bytes = await file.read()
        
        # التعرف على العملة
        result = predict_currency_class(image_bytes)
        
        if result.get("error"):
            logger.error(f"Recognition error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Recognition successful: {result['denomination']} ({result['confidence']:.2f}%)")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/currency_status")
async def currency_status():
    """حالة خدمة التعرف على العملات"""
    model_exists = MODEL_PATH.exists()
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024) if model_exists else 0
    
    return {
        "service": "Currency Recognition",
        "status": "active" if model_exists else "inactive",
        "model_loaded": model_exists,
        "model_size_mb": f"{model_size_mb:.2f}",
        "supported_currencies": SUPPORTED_CURRENCIES,
        "last_check": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
