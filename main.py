from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import joblib
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Saudi Currency Recognition API")

MODEL_PATH = "models/currency/FINAL_SVM_(RBF).pkl"

CURRENCY_NAMES = {
    0: "10 SR",
    1: "100 SR",
    2: "20 SR",
    3: "200 SR",
    4: "5 SR",
    5: "50 SR",
    6: "500 SR"
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

    logger.info(f"SVM classes order: {svm_model.classes_}")
    logger.info("SVM Model loaded!")
    logger.info("=" * 60)
    logger.info("SYSTEM READY!")
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


def predict_currency(image: Image.Image):
    """
    ✅ يرجع (currency_name, confidence) بشكل صحيح
    probabilities مرتبة حسب svm_model.classes_ مو حسب index الـ prediction
    """
    features = extract_features(image)
    prediction = svm_model.predict([features])[0]           # index الفئة (0-6)
    probabilities = svm_model.predict_proba([features])[0]  # مصفوفة مرتبة حسب classes_

    # ✅ الـ confidence = احتمال الفئة المتوقعة بالترتيب الصحيح
    classes = list(svm_model.classes_)
    pred_index_in_proba = classes.index(prediction)
    confidence = float(probabilities[pred_index_in_proba]) * 100

    currency_name = CURRENCY_NAMES.get(prediction, "Unknown")

    logger.info(f"Prediction class: {prediction}, Currency: {currency_name}, Confidence: {confidence:.2f}%")

    return currency_name, round(confidence, 2)



# ──────────────────────────────────────────────
#  /recognize  →  نتيجة واحدة (JSON)
# ──────────────────────────────────────────────
@app.post("/recognize")
async def recognize_currency(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        currency_name, confidence = predict_currency(image)  # ✅

        return JSONResponse({
            "currency": currency_name,
            "confidence": confidence,
            "text_ar": CURRENCY_TEXT_AR.get(currency_name, "عملة غير معروفة"),
            "text_en": currency_name
        })

    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
#  /recognize-with-image  →  نتيجة واحدة (صورة PNG)
# ──────────────────────────────────────────────
@app.post("/recognize-with-image")
async def recognize_with_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        currency_name, confidence = predict_currency(image)  # ✅
        text_ar = CURRENCY_TEXT_AR.get(currency_name, "")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(image)
        ax.axis("off")

        label = f"{currency_name}  ({confidence:.1f}%)\n{text_ar}"
        ax.set_title(label, fontsize=15, color="green",
                     fontweight="bold", pad=12)

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"recognize-with-image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
#  Health check
# ──────────────────────────────────────────────
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "svm_loaded": svm_model is not None,
        "detection_method": "OpenCV",
        "endpoints": [
            "/recognize",
            "/recognize-with-image"
        ]
    }
