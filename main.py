from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys
import cv2
import numpy as np
from datetime import datetime

# Import our currency recognition module
from currency_recognition import (
    initialize_currency_recognition,
    recognize_currency_from_bytes,
    get_currency_recognition_status
)

# ✅ Roboflow for banknote detection
from inference_sdk import InferenceHTTPClient

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
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Roboflow client for banknote detection
ROBOFLOW_API_KEY = "rf_vSnKS4qCBle0NN6iPJzHDVFSg5t1"
roboflow_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global roboflow_client

    logger.info("🔄 Starting model initialization...")
    logger.info("📥 Downloading and loading model...")
    success = initialize_currency_recognition()
    if success:
        logger.info("✅ Model loaded successfully!")
    else:
        logger.error("❌ Failed to load model!")
        sys.exit(1)

    # ✅ Initialize Roboflow client
    try:
        roboflow_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )
        logger.info("✅ Roboflow client initialized!")
    except Exception as e:
        logger.warning(f"⚠️ Roboflow init failed: {e} — multi-detection will use fallback")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Saudi Currency Recognition API",
        "version": "2.1.0",
        "status": "running",
        "endpoints": {
            "recognize": "/recognize (POST) — ورقة واحدة",
            "recognize_multiple": "/recognize-multiple (POST) — عدة أوراق",
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
        "model_status": status,
        "roboflow_ready": roboflow_client is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status")
async def get_status():
    """Get detailed model status"""
    return get_currency_recognition_status()


@app.post("/recognize")
async def recognize_currency(file: UploadFile = File(...)):
    """Recognize a SINGLE Saudi currency note from uploaded image"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        result = recognize_currency_from_bytes(image_bytes)

        return {
            "success": True,
            "currency": result["currency"],
            "confidence": result["confidence"],
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ✅ =============================================
# ✅ NEW: Multiple currency recognition endpoint
# ✅ =============================================
@app.post("/recognize-multiple")
async def recognize_multiple_currencies(file: UploadFile = File(...)):
    """Recognize MULTIPLE Saudi currency notes from a single image"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Decode image for cropping
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # --- Step 1: Detect banknote locations ---
        detections = []

        # Try Roboflow first
        if roboflow_client is not None:
            try:
                detections = detect_with_roboflow(image_bytes)
                logger.info(f"🔍 Roboflow detected {len(detections)} banknote(s)")
            except Exception as e:
                logger.warning(f"⚠️ Roboflow failed: {e}, trying OpenCV fallback")

        # Fallback to OpenCV if Roboflow found nothing or failed
        if len(detections) == 0:
            detections = detect_with_opencv(img)
            logger.info(f"🔍 OpenCV detected {len(detections)} banknote(s)")

        # If still nothing, treat the whole image as one banknote
        if len(detections) == 0:
            logger.info("📄 No detections — treating entire image as single banknote")
            result = recognize_currency_from_bytes(image_bytes)
            return {
                "success": True,
                "count": 1,
                "total": _extract_value(result["currency"]),
                "total_text_ar": f'{_extract_value(result["currency"])} ريال سعودي',
                "total_text_en": f'{_extract_value(result["currency"])} SAR',
                "currencies": [{
                    "currency": result["currency"],
                    "confidence": result["confidence"],
                }],
                "timestamp": datetime.now().isoformat()
            }

        # --- Step 2: Crop each banknote and classify with YOUR model ---
        results = []
        total = 0

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Crop the banknote
            cropped = img[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            # Convert cropped image to bytes
            _, cropped_bytes = cv2.imencode('.jpg', cropped)
            cropped_bytes = cropped_bytes.tobytes()

            # Classify with YOUR existing model
            try:
                result = recognize_currency_from_bytes(cropped_bytes)
                confidence = result["confidence"]

                if confidence > 40:  # Minimum confidence threshold
                    value = _extract_value(result["currency"])
                    total += value
                    results.append({
                        "currency": result["currency"],
                        "confidence": confidence,
                    })
                    logger.info(f"  ✅ {result['currency']} ({confidence:.1f}%)")
                else:
                    logger.info(f"  ⚠️ Low confidence ({confidence:.1f}%), skipping")
            except Exception as e:
                logger.warning(f"  ❌ Classification failed for crop: {e}")

        return {
            "success": True,
            "count": len(results),
            "total": total,
            "total_text_ar": f'{total} ريال سعودي',
            "total_text_en": f'{total} SAR',
            "currencies": results,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-recognition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# =============================================
# Helper functions
# =============================================

def _extract_value(currency_name: str) -> int:
    """Extract numeric value from currency name like '50 SR' → 50"""
    try:
        return int(currency_name.replace("SR", "").strip())
    except (ValueError, AttributeError):
        return 0


def detect_with_roboflow(image_bytes: bytes) -> list:
    """Use Roboflow DOAS Bank Note Detector to find banknote locations"""
    import base64

    # Roboflow expects base64
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')

    result = roboflow_client.infer(img_b64, model_id="bank-note-detector/1")

    detections = []
    for pred in result.get("predictions", []):
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        # Convert center format to corner format
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": pred.get("confidence", 0),
        })

    return detections


def detect_with_opencv(img) -> list:
    """Fallback: Use OpenCV contour detection to find rectangular banknotes"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    img_area = img.shape[0] * img.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter: banknote should be 5%–80% of image area
        if area < img_area * 0.05 or area > img_area * 0.80:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Banknotes are rectangular (4 sides)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h > 0 else 0

            # Banknote aspect ratio ~2:1
            if 1.3 < aspect_ratio < 3.0 or 0.33 < aspect_ratio < 0.77:
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.5,
                })

    # Sort left to right
    detections.sort(key=lambda d: d["bbox"][0])

    return detections


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
