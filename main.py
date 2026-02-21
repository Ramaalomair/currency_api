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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Saudi Currency Recognition API")

MODEL_PATH = "models/currency/FINAL_SVM_(RBF).pkl"

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

mobilenet = None
svm_model = None


@app.on_event("startup")
async def load_model():
    global mobilenet, svm_model
    
    logger.info("=" * 60)
    logger.info("INITIALIZING CURRENCY RECOGNITION SYSTEM")
    logger.info("=" * 60)
    
    logger.info("Loading MobileNetV2 feature extractor...")
    mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
    mobilenet.classifier = torch.nn.Identity()
    mobilenet.eval()
    logger.info("MobileNetV2 loaded (1280-D features)")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    logger.info(f"Loading SVM model: {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        svm_model = joblib.load(f)
    
    logger.info("SVM Model loaded!")
    logger.info("=" * 60)
    logger.info("SYSTEM READY WITH OPENCV DETECTION!")
    logger.info("=" * 60)


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


def detect_banknotes_opencv(img_array: np.ndarray) -> list:
    height, width = img_array.shape[:2]
    
    min_area = (width * height) * 0.005
    max_area = (width * height) * 0.80
    
    logger.info(f"Image size: {width}x{height}")
    logger.info(f"Area range: {min_area:.0f} - {max_area:.0f}")
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"Found {len(contours)} initial contours")
    
    detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        
        if w == 0 or h == 0:
            continue
        
        if h > w:
            w, h = h, w
        
        aspect_ratio = w / h
        
        if 1.3 < aspect_ratio < 4.0:
            x, y, bw, bh = cv2.boundingRect(contour)
            
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + bw + padding)
            y2 = min(height, y + bh + padding)
            
            confidence = max(0.5, 1.0 - (abs(aspect_ratio - 2.2) * 0.15))
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "area": area
            })
            
            logger.info(f"Found: bbox=[{x1},{y1},{x2},{y2}], ratio={aspect_ratio:.2f}")
    
    detections = remove_overlapping_detections(detections, iou_threshold=0.3)
    logger.info(f"Final detections: {len(detections)}")
    
    return detections


def remove_overlapping_detections(detections: list, iou_threshold: float = 0.5) -> list:
    if len(detections) <= 1:
        return detections
    
    detections = sorted(detections, key=lambda x: x["area"], reverse=True)
    kept = []
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        should_keep = True
        
        for kept_det in kept:
            kx1, ky1, kx2, ky2 = kept_det["bbox"]
            
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                union = area1 + area2 - intersection
                
                if intersection / union > iou_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            kept.append(det)
    
    return kept


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
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize-multiple")
async def recognize_multiple_currencies(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        logger.info("Starting multi-currency recognition with OpenCV...")
        
        detections = detect_banknotes_opencv(img_np)
        
        if not detections:
            logger.warning("No banknotes detected, trying full image...")
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
            
            cropped = img_np[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            
            cropped_pil = Image.fromarray(cropped)
            
            features = extract_features(cropped_pil)
            prediction = svm_model.predict([features])[0]
            probabilities = svm_model.predict_proba([features])[0]
            confidence = float(probabilities[prediction]) * 100
            
            logger.info(f"Detection {idx+1}: Class={prediction}, Confidence={confidence:.1f}%")
            
            if confidence < 40:
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
        
        logger.info(f"Final: {len(currencies)} currencies, Total: {total_value} SAR")
        
        return JSONResponse({
            "count": len(currencies),
            "total": total_value,
            "currencies": currencies,
            "total_text_ar": f"المجموع {total_value} ريال سعودي",
            "total_text_en": f"Total {total_value} SAR"
        })
    
    except Exception as e:
        logger.error(f"Multi-recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "svm_loaded": svm_model is not None,
        "detection_method": "OpenCV"
    }
