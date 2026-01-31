import joblib
import numpy as np
from PIL import Image
import io
import os
import sys
import torch
import timm
from torchvision import transforms

# Global variables
MODEL = None
FEATURE_EXTRACTOR = None
DEVICE = torch.device('cpu')
MODEL_PATH = "models/currency/FINAL_SVM_(RBF).pkl"

# Preprocessing transform for MobileNetV2
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
        # 1. Load MobileNetV2 Feature Extractor
        print("📥 Loading MobileNetV2 feature extractor...", file=sys.stderr)
        sys.stderr.flush()
        
        FEATURE_EXTRACTOR = timm.create_model(
            'mobilenetv2_100',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        FEATURE_EXTRACTOR = FEATURE_EXTRACTOR.to(DEVICE)
        FEATURE_EXTRACTOR.eval()
        
        print("✅ MobileNetV2 loaded (Output: 1280-D features)", file=sys.stderr)
        sys.stderr.flush()
        
        # 2. Load SVM Model from local file
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}", file=sys.stderr)
            sys.stderr.flush()
            return False
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model file found locally! ({file_size} bytes)", file=sys.stderr)
        sys.stderr.flush()
        
        print("🔄 Loading SVM model into memory...", file=sys.stderr)
        sys.stderr.flush()
        MODEL = joblib.load(MODEL_PATH)
        print("✅ SVM Model loaded and ready!", file=sys.stderr)
        print(f"   Model type: {type(MODEL)}", file=sys.stderr)
        
        if hasattr(MODEL, 'classes_'):
            print(f"   Classes: {MODEL.classes_}", file=sys.stderr)
        if hasattr(MODEL, 'n_support_'):
            print(f"   Support vectors: {MODEL.n_support_}", file=sys.stderr)
        if hasattr(MODEL, 'probability'):
            print(f"   Probability support: {MODEL.probability}", file=sys.stderr)
        
        sys.stderr.flush()
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return False


def extract_features(image_bytes):
    """Extract 1280-D features using MobileNetV2"""
    global FEATURE_EXTRACTOR
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_tensor = normalize_transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = FEATURE_EXTRACTOR(img_tensor)
        
        features_np = features.cpu().numpy()
        
        print(f"   Features shape: {features_np.shape}", file=sys.stderr)
        sys.stderr.flush()
        
        return features_np
        
    except Exception as e:
        print(f"❌ Error extracting features: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        raise


def recognize_currency_from_bytes(image_bytes):
    """Recognize currency from image bytes using MobileNetV2 + SVM"""
    global MODEL, FEATURE_EXTRACTOR
    
    if MODEL is None or FEATURE_EXTRACTOR is None:
        raise Exception("Models not loaded. Please wait for initialization.")
    
    try:
        print("🔍 Starting currency recognition...", file=sys.stderr)
        sys.stderr.flush()
        
        features = extract_features(image_bytes)
        
        prediction = MODEL.predict(features)
        print(f"   Prediction: {prediction}", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            probabilities = MODEL.predict_proba(features)
            confidence = float(np.max(probabilities) * 100)
            print(f"   Probabilities: {probabilities}", file=sys.stderr)
            print(f"   Confidence: {confidence:.2f}%", file=sys.stderr)
            sys.stderr.flush()
        except AttributeError:
            confidence = 100.0
            print("   (Model doesn't support probability prediction - using 100%)", file=sys.stderr)
            sys.stderr.flush()
        
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
        
        result = {
            "currency": currency_name,
            "confidence": round(confidence, 2),
            "label": currency_label
        }
        
        print(f"✅ Recognition result: {result}", file=sys.stderr)
        sys.stderr.flush()
        return result
        
    except Exception as e:
        print(f"❌ Error during recognition: {str(e)}", file=sys.stderr)
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
        "feature_extractor_loaded": FEATURE_EXTRACTOR is not None
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
        except:
            pass
    
    return status


def currency_recognizer():
    """For backward compatibility"""
    return MODEL
