from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import logging
import threading
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
    "5 SR":   "خمسة ريالات سعودية",
    "10 SR":  "عشرة ريالات سعودية",
    "20 SR":  "عشرون ريالاً سعودياً",
    "50 SR":  "خمسون ريالاً سعودياً",
    "100 SR": "مئة ريال سعودي",
    "200 SR": "مئتا ريال سعودي",
    "500 SR": "خمسمئة ريال سعودي",
}

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

feature_extractor = None
svm_model = None
models_ready = False


def load_models_background():
    global feature_extractor, svm_model, models_ready
    try:
        logger.info("=" * 60)
        logger.info("LOADING MODELS IN BACKGROUND...")
        logger.info("=" * 60)

        logger.info("Loading MobileNetV2...")
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
        feature_extractor = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        feature_extractor.eval()
        logger.info("✅ MobileNetV2 loaded")

        logger.info("Loading SVM model...")
        with open(MODEL_PATH, 'rb') as f:
            svm_model = joblib.load(f)
        logger.info(f"✅ SVM loaded — classes: {svm_model.classes_}")

        models_ready = True
        logger.info("=" * 60)
        logger.info("✅ ALL MODELS READY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")


@app.on_event("startup")
async def startup():
    logger.info("🚀 Server starting — models loading in background...")
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()


def extract_features(image: Image.Image) -> np.ndarray:
    img_tensor = PREPROCESS(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.view(features.size(0), -1).squeeze().numpy()


def predict_currency(image: Image.Image):
    features = extract_features(image)
    prediction = svm_model.predict([features])[0]
    probabilities = svm_model.predict_proba([features])[0]

    classes = list(svm_model.classes_)
    pred_index_in_proba = classes.index(prediction)
    confidence = float(probabilities[pred_index_in_proba]) * 100

    currency_name = CURRENCY_NAMES.get(prediction, "Unknown")

    logger.info("All probabilities:")
    for cls_idx, prob in zip(classes, probabilities):
        cls_name = CURRENCY_NAMES.get(cls_idx, f"?{cls_idx}")
        logger.info(f"  {cls_name}: {prob*100:.2f}%")

    logger.info(f"✅ Prediction: {currency_name} ({confidence:.2f}%)")
    return currency_name, round(confidence, 2)


# ──────────────────────────────────────────────
#  /recognize
# ──────────────────────────────────────────────
@app.post("/recognize")
async def recognize_currency(file: UploadFile = File(...)):
    if not models_ready:
        raise HTTPException(
            status_code=503,
            detail="Models still loading, please try again in a moment"
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        currency_name, confidence = predict_currency(image)
        return JSONResponse({
            "currency":   currency_name,
            "confidence": confidence,
            "text_ar":    CURRENCY_TEXT_AR.get(currency_name, "عملة غير معروفة"),
            "text_en":    currency_name
        })
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
#  /recognize-with-image
# ──────────────────────────────────────────────
@app.post("/recognize-with-image")
async def recognize_with_image(file: UploadFile = File(...)):
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        currency_name, confidence = predict_currency(image)
        text_ar = CURRENCY_TEXT_AR.get(currency_name, "")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(image)
        ax.axis("off")
        label = f"{currency_name}  ({confidence:.1f}%)\n{text_ar}"
        ax.set_title(label, fontsize=15, color="green", fontweight="bold", pad=12)

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
#  Health check
# ──────────────────────────────────────────────
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status":       "healthy" if models_ready else "loading",
        "models_ready": models_ready,
        "svm_loaded":   svm_model is not None,
        "preprocessing": "Resize(256) → CenterCrop(224) → Normalize",
        "feature_dim":  "1280-D (MobileNetV2.features + AdaptiveAvgPool2d)",
        "endpoints":    ["/recognize", "/recognize-with-image"]
    }
