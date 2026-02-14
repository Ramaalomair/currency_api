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
from typing import List, Dict
import os
from inference_sdk import InferenceHTTPClient

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

MODEL_PATH = "models/mobilenetv2_svm_saudi_currency.pkl"

# ✅ ضعي مفتاح Roboflow هنا
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
ROBOFLOW_MODEL_ID = "saudi-banknotes/1"  # أو أي model ID عندك

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
# GLOBAL MODEL
# ========================================
mobilenet = None
svm_model = None
roboflow_client = None

# ========================================
# STARTUP
# ========================================
@app.on_event("startup")
async def load_model():
    global mobilenet, svm_model, roboflow_client
    
    logger.info("🔄 Starting model initialization...")
    logger.info("=" * 60)
    logger.info("🔄 INITIALIZING CURRENCY RECOGNITION MODEL")
    logger.info("=" * 60)
    
    # MobileNetV2
    logger.info("📥 Loading MobileNetV2 feature extractor...")
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier = torch.nn.Identity()
    mobilenet.eval()
    logger.info("✅ MobileNetV2 loaded (Output: 1280-D features)")
    
    # SVM
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    logger.info(f"✅ Model file found! ({os.path.getsize(MODEL_PATH)} bytes)")
    logger.info("🔄 Loading SVM model into memory...")
    
    with open(MODEL_PATH, 'rb') as f:
        svm_model = joblib.load(f)
    
    logger.info("✅ SVM Model loaded and ready!")
    logger.info(f"   Model type: {type(svm_model)}")
    logger.info(f"   Classes: {svm_model.classes_}")
    logger.info(f"   Support vectors: {svm_model.n_support_}")
    logger.info(f"   Probability support: {svm_model.probability}")
    
    # Roboflow
    if ROBOFLOW_API_KEY and ROBOFLOW_API_KEY != "YOUR_API_KEY_HERE":
        try:
            logger.info("🔄 Initializing Roboflow client...")
            roboflow_client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=ROBOFLOW_API_KEY
            )
            logger.info("✅ Roboflow client initialized!")
        except Exception as e:
            logger.warning(f"⚠️ Roboflow init failed: {e}")
            roboflow_client = None
    else:
        logger.warning("⚠️ Roboflow API key not set - multi-detection will be limited")

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
def detect_with_roboflow(img_bytes: bytes) -> list:
    """Use Roboflow to detect multiple banknotes"""
    if roboflow_client is None:
        logger.warning("⚠️ Roboflow client not initialized")
        return []
    
    try:
        result = roboflow_client.infer(img_bytes, model_id=ROBOFLOW_MODEL_ID)
        
        detections = []
        if "predictions" in result:
            for pred in result["predictions"]:
                x_center = pred["x"]
                y_center = pred["y"]
                width = pred["width"]
                height = pred["height"]
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": pred.get("confidence", 0.5)
                })
        
        logger.info(f"📊 Roboflow detected {len(detections)} banknotes")
        return detections
    
    except Exception as e:
        logger.error(f"❌ Roboflow detection failed: {e}")
        return []

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
        
        # ✅ استخدم Roboflow للكشف
        detections = detect_with_roboflow(contents)
        
        if not detections:
            logger.warning("⚠️ No banknotes detected by Roboflow")
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
            
            # تأكد من الحدود
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_np.shape[1], x2)
            y2 = min(img_np.shape[0], y2)
            
            # Crop
            cropped = img_np[y1:y2, x1:x2]
            if cropped.size == 0:
                logger.warning(f"⚠️ Detection {idx+1}: Empty crop, skipping")
                continue
            
            cropped_pil = Image.fromarray(cropped)
            
            # Classify
            features = extract_features(cropped_pil)
            prediction = svm_model.predict([features])[0]
            probabilities = svm_model.predict_proba([features])[0]
            confidence = float(probabilities[prediction]) * 100
            
            logger.info(f"💵 Detection {idx+1}: Class {prediction}, Confidence {confidence:.1f}%")
            
            # فلتر منخفض الثقة
            if confidence < 35:
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
        
        logger.info(f"✅ Final result: {len(currencies)} currencies, Total: {total_value} SAR")
        
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
        "model": "loaded",
        "roboflow": "enabled" if roboflow_client else "disabled"
    }
