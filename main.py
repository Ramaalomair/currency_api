# main.py - Munir Currency Recognition API
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

# Currency Recognition Module
from currency_recognition import (
    initialize_currency_recognition,
    recognize_currency_from_bytes,
    get_currency_recognition_status
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Munir Currency Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Currency Recognition Models
# ============================================================
logger.info("⏳ Loading Currency Recognition models...")
try:
    currency_loaded = initialize_currency_recognition()
    if currency_loaded:
        logger.info("✅ Currency Recognition models loaded successfully!")
    else:
        logger.warning("⚠️ Currency Recognition models failed to load")
except Exception as e:
    logger.error(f"❌ Currency Recognition loading failed: {e}")
    currency_loaded = False

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def root():
    """Root endpoint - API status"""
    currency_status = get_currency_recognition_status()
    return {
        "api": "Munir Currency Recognition API",
        "version": "1.0.0",
        "status": "running",
        "currency_recognition": "loaded" if currency_status['loaded'] else "not loaded",
        "supported_currencies": currency_status['classes'] if currency_status['loaded'] else [],
        "languages": ["arabic", "english"]
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    currency_status = get_currency_recognition_status()
    return {
        "status": "healthy",
        "currency_recognition": currency_status['loaded'],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recognize_currency")
async def recognize_currency(file: UploadFile = File(...)):
    """
    Recognize Saudi Riyal banknote denomination from image
    
    Args:
        file: Image file containing a banknote
        
    Returns:
        Recognition result with currency value and voice guidance text (Arabic + English)
    """
    try:
        # Check if currency recognition is loaded
        status = get_currency_recognition_status()
        if not status['loaded']:
            raise HTTPException(503, "Currency recognition service not ready")
        
        # Read image
        img_bytes = await file.read()
        
        # Recognize currency
        result = recognize_currency_from_bytes(img_bytes)
        
        if not result['success']:
            raise HTTPException(400, result.get('message', 'Recognition failed'))
        
        logger.info(f"✅ Currency recognized: {result['currency']} ({result['confidence_percent']}%)")
        
        return result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Currency recognition error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Currency recognition failed: {str(e)}")

@app.get("/currency_status")
def currency_status():
    """Get currency recognition service status"""
    try:
        status = get_currency_recognition_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        raise HTTPException(500, f"Status check failed: {str(e)}")

# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Munir Currency Recognition API...")
    logger.info("   - Saudi Riyal Recognition: 7 denominations")
    logger.info("   - Languages: Arabic + English")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
