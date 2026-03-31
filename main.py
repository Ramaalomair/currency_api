import os
os.environ["U2NET_HOME"] = "/root/.u2net"

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
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rembg import remove, new_session

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

# ✅ نفس الـ preprocessing اللي اتدرب عليه
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

feature_extractor = None
svm_model = None
rembg_session = None


@app.on_event("startup")
async def load_model():
    global feature_extractor, svm_model, rembg_session

    logger.info("=" * 60)
    logger.info("INITIALIZING CURRENCY RECOGNITION SYSTEM")
    logger.info("=" * 60)

    # ✅ MobileNetV2 feature extractor
    logger.info("Loading MobileNetV2 feature extractor...")
    mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
    feature_extractor = nn.Sequential(
        mobilenet.features,
        nn.AdaptiveAvgPool2d((1, 1))
    )
    feature_extractor.eval()
    logger.info("✅ MobileNetV2 loaded — Output: 1280-D features")

    # ✅ rembg session — يحمّل مرة وحدة عند الـ startup
    logger.info("Loading rembg background removal model...")
    rembg_session = new_session("u2net")
    logger.info("✅ rembg loaded!")

    # Load SVM
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    logger.info(f"Loading SVM model: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        svm_model = joblib.load(f)

    logger.info(f"✅ SVM loaded — classes: {svm_model.classes_}")
    logger.info("=" * 60)
    logger.info("SYSTEM READY!")
    logger.info("=" * 60)


def remove_background(image: Image.Image) -> Image.Image:
    """
    شيل الخلفية وحط خلفية بيضاء
    عشان اليد والخلفية ما تأثر على الـ model
    """
    try:
        output = remove(image, session=rembg_session)
        background = Image.new("RGB", output.size, (255, 255, 255))
        output_rgba = output.convert("RGBA")
        background.paste(output_rgba, mask=output_rgba.split()[3])
        logger.info("✅ Background removed successfully")
        return background
    except Exception as e:
        logger.warning(f"⚠️ Background removal failed: {e} — using original")
        return image


def extract_features(image: Image.Image) -> np.ndarray:
    """
    1. شيل الخلفية
    2. Resize(256) → CenterCrop(224) → Normalize
    3. MobileNetV2.features + AdaptiveAvgPool2d → 1280-D
    """
    image = remove_background(image)
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
        logger.error(f"recognize-with-image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
#  Health check
# ──────────────────────────────────────────────
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status":           "healthy",
        "svm_loaded":       svm_model is not None,
        "extractor_loaded": feature_extractor is not None,
        "rembg_loaded":     rembg_session is not None,
        "preprocessing":    "rembg → Resize(256) → CenterCrop(224) → Normalize",
        "feature_dim":      "1280-D (MobileNetV2.features + AdaptiveAvgPool2d)",
        "endpoints":        ["/recognize", "/recognize-with-image"]
    }
