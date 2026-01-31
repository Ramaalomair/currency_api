from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys

# Import our currency recognition module
from currency_recognition import (
    initialize_currency_recognition,
    recognize_currency_from_bytes,
    get_currency_recognition_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Saudi Currency Recognition API",
    description="API for recognizing Saudi Riyal banknotes using MobileNetV2 + SVM",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("🔄 Starting model initialization...")
    logger.info("📥 Downloading and loading model...")
    success = initialize_currency_recognition()
    if success:
        logger.info("✅ Model loaded successfully!")
    else:
        logger.error("❌ Failed to load model!")
        sys.exit(1)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Saudi Currency Recognition API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Confidence threshold filtering",
            "Better error messages",
            "Support for low-confidence detection"
        ],
        "endpoints": {
            "recognize": "/recognize (POST)",
            "status": "/status (GET)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = get_currency_recognition_status()
    return {
        "status": "healthy" if status["initialized"] else "unhealthy",
        "model_loaded": status["initialized"],
        "confidence_threshold": status.get("confidence_threshold", 60.0)
    }


@app.get("/status")
async def get_status():
    """Get detailed model status"""
    return get_currency_recognition_status()


@app.post("/recognize")
async def recognize_currency(file: UploadFile = File(...)):
    """
    Recognize Saudi currency from uploaded image
    
    الآن مع فلتر الـ confidence:
    - إذا confidence < 60% → يرجع خطأ مع رسالة واضحة
    - إذا confidence >= 60% → يرجع النتيجة
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Recognize currency
        result = recognize_currency_from_bytes(image_bytes)
        
        # ====== Handle low confidence case ======
        if not result.get("success", True):
            # Return 200 OK but with error info
            # (بعض المطورين يفضلون 200 مع success: false)
            # إذا تبي 400 Bad Request، غيّر JSONResponse إلى HTTPException
            return JSONResponse(
                status_code=200,  # أو 400 إذا تبي
                content={
                    "success": False,
                    "error": result.get("error"),
                    "message_ar": result.get("message"),
                    "message_en": result.get("message_en"),
                    "confidence": result.get("confidence"),
                    "suggested_currency": result.get("suggested_currency"),
                    "threshold": result.get("threshold"),
                    "tip": "حاول التقاط الصورة في إضاءة جيدة وتأكد من وضوح العملة"
                }
            )
        
        # ====== Success case ======
        return {
            "success": True,
            "currency": result["currency"],
            "confidence": result["confidence"],
            "label": result["label"],
            "message": f"تم التعرف على العملة: {result['currency']} بنسبة ثقة {result['confidence']:.1f}%"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MAIN.PY STARTING...")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10000,
        log_level="info"
    )
