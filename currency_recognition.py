import joblib
import numpy as np
from PIL import Image
import io
import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms

# Global variables
MODEL = None
FEATURE_EXTRACTOR = None
DEVICE = torch.device('cpu')
MODEL_PATH = "models/currency/FINAL_SVM_(RBF).pkl"

# MINIMUM CONFIDENCE THRESHOLD
MIN_CONFIDENCE_THRESHOLD = 60.0

# Preprocessing (EXACTLY as in training notebook)
normalize_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def initialize_currency_recognition():
    """Load SVM model and MobileNetV2 feature extractor"""
    global MODEL, FEATURE_EXTRACTOR
    
    print("=" * 60, file=sys.stderr)
    print("🔄 INITIALIZING CURRENCY RECOGNITION MODEL", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.stderr.flush()
    
    if MODEL is not None and FEATURE_EXTRACTOR is not None:
        print("✅ Models already loaded!", file=sys.stderr)
        sys.stderr.flush()
        return True
    
    try:
        # ====== CRITICAL: Use EXACT same method as training ======
        print("📥 Loading MobileNetV2 (torchvision method)...", file=sys.stderr)
        sys.stderr.flush()
        
        # Load MobileNetV2 using torchvision (NOT timm)
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Extract only the features part
        FEATURE_EXTRACTOR = mobilenet.features
        
        # Add AdaptiveAvgPool2d to get (batch, 1280, 1, 1)
        FEATURE_EXTRACTOR = nn.Sequential(
            FEATURE_EXTRACTOR,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        FEATURE_EXTRACTOR = FEATURE_EXTRACTOR.to(DEVICE)
        FEATURE_EXTRACTOR.eval()
        
        print("✅ MobileNetV2 loaded (torchvision)", file=sys.stderr)
        print("   Output: 1280-D features", file=sys.stderr)
        sys.stderr.flush()
        
        # Load SVM Model
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}", file=sys.stderr)
            sys.stderr.flush()
            return False
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model file found! ({file_size} bytes)", file=sys.stderr)
        sys.stderr.flush()
        
        print("🔄 Loading SVM model...", file=sys.stderr)
        sys.stderr.flush()
        MODEL = joblib.load(MODEL_PATH)
        print("✅ SVM Model loaded!", file=sys.stderr)
        print(f"   Type: {type(MODEL)}", file=sys.stderr)
        
        if hasattr(MODEL, 'classes_'):
            print(f"   Classes: {MODEL.classes_}", file=sys.stderr)
        if hasattr(MODEL, 'n_support_'):
            print(f"   Support vectors: {MODEL.n_support_}", file=sys.stderr)
        if hasattr(MODEL, 'probability'):
            print(f"   Probability: {MODEL.probability}", file=sys.stderr)
        if hasattr(MODEL, 'gamma'):
            print(f"   Gamma: {MODEL.gamma}", file=sys.stderr)
        if hasattr(MODEL, 'C'):
            print(f"   C: {MODEL.C}", file=sys.stderr)
        
        sys.stderr.flush()
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return False


def extract_features(image_bytes):
    """Extract 1280-D features using MobileNetV2 (EXACTLY as in training)"""
    global FEATURE_EXTRACTOR
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        print(f"   📸 Image: size={img.size}, mode={img.mode}", file=sys.stderr)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply preprocessing (same as training)
        img_tensor = normalize_transform(img).unsqueeze(0).to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            features = FEATURE_EXTRACTOR(img_tensor)
        
        # Flatten: (1, 1280, 1, 1) -> (1, 1280)
        features = features.view(features.size(0), -1)
        
        features_np = features.cpu().numpy()
        
        print(f"   ✅ Features: shape={features_np.shape}", file=sys.stderr)
        sys.stderr.flush()
        
        return features_np
        
    except Exception as e:
        print(f"❌ Error extracting features: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        raise


def recognize_currency_from_bytes(image_bytes):
    """
    Recognize currency from image bytes using MobileNetV2 + SVM
    
    Now with confidence threshold filtering
    """
    global MODEL, FEATURE_EXTRACTOR
    
    if MODEL is None or FEATURE_EXTRACTOR is None:
        raise Exception("Models not loaded. Please wait for initialization.")
    
    try:
        print("🔍 Starting currency recognition...", file=sys.stderr)
        sys.stderr.flush()
        
        # Extract features
        features = extract_features(image_bytes)
        
        # Get prediction
        prediction = MODEL.predict(features)
        print(f"   🎯 Prediction: {prediction[0]}", file=sys.stderr)
        
        # Get probabilities
        try:
            probabilities = MODEL.predict_proba(features)[0]
            confidence = float(np.max(probabilities) * 100)
            
            # Print all probabilities
            currencies_short = ["10SR", "100SR", "20SR", "200SR", "5SR", "50SR", "500SR"]
            print(f"   📊 Probabilities:", file=sys.stderr)
            for curr, prob in sorted(zip(currencies_short, probabilities), 
                                    key=lambda x: x[1], reverse=True):
                print(f"      {curr:6} : {prob*100:5.2f}%", file=sys.stderr)
            
            print(f"   📈 Confidence: {confidence:.2f}%", file=sys.stderr)
            sys.stderr.flush()
            
        except AttributeError:
            confidence = 100.0
            print("   ⚠️ Model doesn't support probability", file=sys.stderr)
            sys.stderr.flush()
        
        # Currency mapping
        currencies = {
            0: "10 SR",
            1: "100 SR",
            2: "20 SR",
            3: "200 SR",
            4: "5 SR",
            5: "50 SR",
            6: "500 SR"
        }
        
        currency_label = int(prediction[0])
        currency_name = currencies.get(currency_label, f"Unknown (Label: {currency_label})")
        
        # Check confidence threshold
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            print(f"   ❌ REJECTED: {confidence:.2f}% < {MIN_CONFIDENCE_THRESHOLD}%", file=sys.stderr)
            sys.stderr.flush()
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3 = [(currencies[i], probabilities[i]*100) for i in top_3_indices]
            
            result = {
                "success": False,
                "error": "low_confidence",
                "message": "الصورة غير واضحة. الرجاء التقاط صورة أفضل للعملة",
                "message_en": "Image not clear. Please take a better photo of the currency",
                "confidence": round(confidence, 2),
                "predicted": currency_name,
                "top_3_predictions": [
                    {"currency": curr, "confidence": round(conf, 2)}
                    for curr, conf in top_3
                ],
                "threshold": MIN_CONFIDENCE_THRESHOLD,
                "advice_ar": "نصائح: 1) إضاءة جيدة 2) تركيز واضح 3) بدون ظلال",
                "advice_en": "Tips: 1) Good lighting 2) Clear focus 3) No shadows"
            }
            
            return result
        
        # Success
        print(f"   ✅ ACCEPTED: {confidence:.2f}% >= {MIN_CONFIDENCE_THRESHOLD}%", file=sys.stderr)
        sys.stderr.flush()
        
        result = {
            "success": True,
            "currency": currency_name,
            "confidence": round(confidence, 2),
            "label": currency_label
        }
        
        print(f"   🎉 Result: {result}", file=sys.stderr)
        sys.stderr.flush()
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


def get_currency_recognition_status():
    """Get model status"""
    status = {
        "initialized": MODEL is not None and FEATURE_EXTRACTOR is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "feature_extractor_loaded": FEATURE_EXTRACTOR is not None,
        "feature_extractor_type": "torchvision.models.mobilenet_v2",
        "confidence_threshold": MIN_CONFIDENCE_THRESHOLD
    }
    
    if MODEL is not None:
        status["model_type"] = str(type(MODEL))
        
        try:
            if hasattr(MODEL, 'n_support_'):
                status["n_support_vectors"] = MODEL.n_support_.tolist()
            if hasattr(MODEL, 'classes_'):
                status["classes"] = MODEL.classes_.tolist()
            if hasattr(MODEL, 'kernel'):
                status["kernel"] = MODEL.kernel
            if hasattr(MODEL, 'gamma'):
                status["gamma"] = str(MODEL.gamma)
            if hasattr(MODEL, 'C'):
                status["C"] = float(MODEL.C)
        except:
            pass
    
    return status


def currency_recognizer():
    """For backward compatibility"""
    return MODEL
