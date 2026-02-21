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
# OPENCV BANKNOTE DETECTION
# ========================================
def detect_banknotes_opencv(img_array: np.ndarray) -> list:
    """Detect banknotes using OpenCV contour detection"""
    
    original = img_array.copy()
    height, width = img_array.shape[:2]
    
    # حساب الحد الأدنى للمساحة بناءً على حجم الصورة
    min_area = (width * height) * 0.01  # 1% من الصورة
    max_area = (width * height) * 0.95  # 95% من الصورة
    
    logger.info(f"📐 Image size: {width}x{height}")
    logger.info(f"📐 Area range: {min_area:.0f} - {max_area:.0f}")
    
    # تحويل لـ grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # تقليل الـ noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # تحسين الحواف
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # إيجاد الـ contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"🔍 Found {len(contours)} initial contours")
    
    detections = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # فلترة حسب المساحة
        if area < min_area or area > max_area:
            continue
        
        # تقريب الشكل
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # الحصول على المستطيل المحيط
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # حساب الأبعاد
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
            
        # التأكد إن العرض أكبر من الطول
        if h > w:
            w, h = h, w
        
        aspect_ratio = w / h
        
        # نسبة العملة السعودية تقريباً 2.2:1
        if 1.5 < aspect_ratio < 3.5:
            # الحصول على bounding box عادي
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # إضافة padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + bw + padding)
            y2 = min(height, y + bh + padding)
            
            # حساب confidence بناءً على مدى قرب النسبة من 2.2
            ideal_ratio = 2.2
            ratio_diff = abs(aspect_ratio - ideal_ratio)
            confidence = max(0.5, 1.0 - (ratio_diff * 0.2))
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "aspect_ratio": aspect_ratio,
                "area": area
            })
            
            logger.info(f"✅ Detection {len(detections)}: bbox=[{x1},{y1},{x2},{y2}], ratio={aspect_ratio:.2f}, conf={confidence:.2f}")
    
    # إزالة التكرارات المتداخلة
    detections = remove_overlapping_detections(detections)
    
    logger.info(f"📊 Final detections after filtering: {len(detections)}")
    
    return detections


def remove_overlapping_detections(detections: list, iou_threshold: float = 0.5) -> list:
    """إزالة الـ detections المتداخلة"""
    
    if len(detections) <= 1:
        return detections
    
    # ترتيب حسب المساحة (الأكبر أولاً)
    detections = sorted(detections, key=lambda x: x["area"], reverse=True)
    
    kept = []
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        
        should_keep = True
        
        for kept_det in kept:
            kx1, ky1, kx2, ky2 = kept_det["bbox"]
            
            # حساب التداخل
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            kept.append(det)
    
    return kept


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
            logger.warning("⚠️ No banknotes detected, trying full image...")
            # لو ما لقى شي، جرب الصورة كاملة
            features = extract_features(image)
            prediction = svm_model.predict([features])[0]
            probabilities = svm_model.predict_proba([features])[0]
            confidence = float(probabilities[prediction]) * 100
            
            if confidence >= 40:
                currency_name = CURRENCY_NAMES.get(prediction, "Unknown")
                value = int(currency_name.replace(" SR", "")) if currency_name != "Unknown" else 0
                
                return JSONResponse({
                    "count": 1,
                    "total": value,
                    "currencies": [{
                        "currency": currency_name,
                        "confidence": round(confidence, 2),
                        "text_ar": CURRENCY_TEXT_AR.get(currency_name, ""),
                        "text_en": currency_name
                    }],
                    "total_text_ar": f"المجموع {value} ريال سعودي",
                    "total_text_en": f"Total {value} SAR"
                })
            
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
                logger.info(f"⚠️ Detection {idx+1}: Low confidence ({confidence:.1f}%), skipping")
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
    }    4: "100 SR", 5: "200 SR", 6: "500 SR"
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
    logger.info("✅ SYSTEM READY WITH ROBOFLOW DETECTION!")
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
# ROBOFLOW DETECTION
# ========================================
def detect_banknotes_roboflow(img_array: np.ndarray) -> list:
    """Detect banknotes using Roboflow hosted API"""
    
    # Convert to base64
    image_pil = Image.fromarray(img_array)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Call Roboflow API
    response = requests.post(
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL}",
        params={"api_key": ROBOFLOW_API_KEY},
        data=img_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    result = response.json()
    logger.info(f"🔍 Roboflow response: {result}")
    detections = []
    
    for pred in result.get("predictions", []):
        x_center = pred["x"]
        y_center = pred["y"]
        w = pred["width"]
        h = pred["height"]
        
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": pred["confidence"]
        })
    
    logger.info(f"📊 Roboflow detected {len(detections)} banknotes")
    return detections

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
# MULTI RECOGNITION WITH ROBOFLOW
# ========================================
@app.post("/recognize-multiple")
async def recognize_multiple_currencies(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        logger.info("🔍 Starting multi-currency recognition with Roboflow...")
        
        # Detect with Roboflow
        detections = detect_banknotes_roboflow(img_np)
        
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
        "detection_method": "Roboflow"
    }
