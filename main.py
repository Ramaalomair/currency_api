from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import joblib
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import logging
import os
import cv2

# ========================================
# LOGGING
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# APP & CONFIG
# ========================================
app = FastAPI(title="Saudi Currency Recognition API")

MODEL_PATH = "models/currency/FINAL_SVM_(RBF).pkl"

# ========================================
# CURRENCY MAPPING
# ========================================
CURRENCY_NAMES = {
    0: "5 SR", 1: "10 SR", 2: "20 SR", 3: "50 SR",
    4: "100 SR", 5: "200 SR", 6: "500 SR"
}

CURRENCY_TEXT_AR = {
    "5 SR": "خمسة ريالات سعودية",
    "10 SR": "عشرة ريالات سعودية",
    "20 SR": "عشرون ريالاً سعودياً",
    "50 SR": "خمسون ريالاً سعودياً",
    "100 SR": "مئة ريال سعودي",
    "200 SR": "مئتا ريال سعودي",
    "500 SR": "خمسمئة ريال سعودي",
}

# ========================================
# GLOBAL MODELS
# ========================================
mobilenet = None
svm_model = None

# ========================================
# STARTUP
# ========================================
@app.on_event("startup")
async def load_model():
    global mobilenet, svm_model
    
    logger.info("=" * 60)
    logger.info("🔄 INITIALIZING CURRENCY RECOGNITION SYSTEM")
    logger.info("=" * 60)
    
    # MobileNetV2
    logger.info("📥 Loading MobileNetV2 feature extractor...")
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier = torch.nn.Identity()
    mobilenet.eval()
    logger.info("✅ MobileNetV2 loaded (1280-D features)")
    
    # SVM
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    logger.info(f"📥 Loading SVM model: {MODEL_PATH}")
    logger.info(f"   File size: {os.path.getsize(MODEL_PATH)} bytes")
    
    with open(MODEL_PATH, 'rb') as f:
        svm_model = joblib.load(f)
    
    logger.info("✅ SVM Model loaded!")
    logger.info(f"   Model type: {type(svm_model)}")
    logger.info(f"   Classes: {svm_model.classes_}")
    if hasattr(svm_model, 'n_support_'):
        logger.info(f"   Support vectors: {svm_model.n_support_}")
    
    logger.info("=" * 60)
    logger.info("✅ SYSTEM READY WITH OPENCV DETECTION!")
    logger.info("=" * 60)

# ========================================
# FEATURE EXTRACTION
# ========================================
def extract_features(image: Image.Image) -> np.ndarray:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        features = mobilenet(img_tensor)
    
    return features.squeeze().numpy()

# ========================================
# OPENCV DETECTION HELPERS
# ========================================
def compute_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1, y1, x2, y2 = box1
    fx1, fy1, fx2, fy2 = box2
    
    ix1 = max(x1, fx1)
    iy1 = max(y1, fy1)
    ix2 = min(x2, fx2)
    iy2 = min(y2, fy2)
    
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (fx2 - fx1) * (fy2 - fy1)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

# ========================================
# ENHANCED OPENCV DETECTION
# ========================================
def detect_banknotes_opencv(img_array: np.ndarray) -> list:
    """Enhanced OpenCV detection for multiple banknotes"""
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Strong contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    all_detections = []
    img_area = img_array.shape[0] * img_array.shape[1]
    
    # Multiple edge detection configurations
    configs = [
        (3, 15, 50),   # (blur_kernel, canny_low, canny_high)
        (5, 20, 70),
        (5, 25, 90),
        (7, 30, 100),
        (9, 20, 80),
    ]
    
    for blur_k, canny_l, canny_h in configs:
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred, canny_l, canny_h)
        
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filter: 1% - 95% of image
            if area < img_area * 0.01 or area > img_area * 0.95:
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h if h > 0 else 0
                
                # Wide aspect ratio range
                if 0.15 < aspect_ratio < 6.0:
                    all_detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "area": area
                    })
    
    # Remove duplicates using IoU
    unique_detections = []
    for det in all_detections:
        is_duplicate = False
        for unique_det in unique_detections:
            iou = compute_iou(det["bbox"], unique_det["bbox"])
            if iou > 0.3:  # 30% overlap = duplicate
                is_duplicate = True
                # Keep the larger one
                if det["area"] > unique_det["area"]:
                    unique_detections.remove(unique_det)
                    is_duplicate = False
                break
        
        if not is_duplicate:
            unique_detections.append(det)
    
    # Sort left to right
    unique_detections.sort(key=lambda d: d["bbox"][0])
    
    logger.info(f"📊 OpenCV detected {len(unique_detections)} banknotes")
    
    # Return in expected format
    return [{"bbox": d["bbox"], "confidence": 0.5} for d in unique_detections]

# ========================================
# SINGLE RECOGNITION
# ========================================
@app.post("/recognize")
async def recognize_currency(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        features = extract_features(image)
        prediction = svm_model.predict([features])[0]
        probabilities = svm_model.predict_proba([features])[0]
        confidence = float(probabilities[prediction]) * 100
        
        currency_name = CURRENCY_NAMES.get(prediction, "Unknown")
        
        return JSONResponse({
            "currency": currency_name,
            "confidence": round(confidence, 2),
            "text_ar": CURRENCY_TEXT_AR.get(currency_name, "عملة غير معروفة"),
            "text_en": currency_name
        })
    
    except Exception as e:
        logger.error(f"❌ Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# MULTI RECOGNITION WITH OPENCV
# ========================================
@app.post("/recognize-multiple")
async def recognize_multiple_currencies(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        logger.info("🔍 Starting multi-currency recognition with OpenCV...")
        
        # Detect with OpenCV
        detections = detect_banknotes_opencv(img_np)
        
        if not detections:
            logger.warning("⚠️ No banknotes detected")
            return JSONResponse({
                "count": 0,
                "total": 0,
                "currencies": [],
                "total_text_ar": "لم يتم اكتشاف أي عملة",
                "total_text_en": "No currency detected"
            })
        
        currencies = []
        total_value = 0
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            
            # Ensure bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_np.shape[1], x2)
            y2 = min(img_np.shape[0], y2)
            
            # Crop banknote
            cropped = img_np[y1:y2, x1:x2]
            if cropped.size == 0:
                logger.warning(f"⚠️ Detection {idx+1}: Empty crop, skipping")
                continue
            
            cropped_pil = Image.fromarray(cropped)
            
            # Classify with SVM
            features = extract_features(cropped_pil)
            prediction = svm_model.predict([features])[0]
            probabilities = svm_model.predict_proba([features])[0]
            confidence = float(probabilities[prediction]) * 100
            
            logger.info(f"💵 Detection {idx+1}: Class={prediction}, Confidence={confidence:.1f}%")
            
            # Filter low confidence
            if confidence < 40:
                logger.info(f"⚠️ Detection {idx+1}: Low confidence, skipping")
                continue
            
            currency_name = CURRENCY_NAMES.get(prediction, "Unknown")
            value = int(currency_name.replace(" SR", "")) if currency_name != "Unknown" else 0
            
            currencies.append({
                "currency": currency_name,
                "confidence": round(confidence, 2),
                "text_ar": CURRENCY_TEXT_AR.get(currency_name, ""),
                "text_en": currency_name
            })
            
            total_value += value
        
        logger.info(f"✅ Final: {len(currencies)} currencies, Total: {total_value} SAR")
        
        return JSONResponse({
            "count": len(currencies),
            "total": total_value,
            "currencies": currencies,
            "total_text_ar": f"المجموع {total_value} ريال سعودي",
            "total_text_en": f"Total {total_value} SAR"
        })
    
    except Exception as e:
        logger.error(f"❌ Multi-recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# HEALTH CHECK
# ========================================
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "svm_loaded": svm_model is not None,
        "detection_method": "OpenCV"
    }
