import joblib
import numpy as np
from PIL import Image
import io
import os
import sys
import torch
import timm
from torchvision import transforms
from datetime import datetime

# Global variables
MODEL = None
FEATURE_EXTRACTOR = None
DEVICE = torch.device('cpu')
MODEL_PATH = "models/currency/FINAL_SVM_(RBF).pkl"
DEBUG_DIR = "debug_images"  # مجلد لحفظ الصور للفحص

# Create debug directory
os.makedirs(DEBUG_DIR, exist_ok=True)

# MINIMUM CONFIDENCE THRESHOLD
MIN_CONFIDENCE_THRESHOLD = 60.0

# Preprocessing (EXACTLY as in training)
normalize_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def save_debug_image(img, prefix="original"):
    """Save image for debugging"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{DEBUG_DIR}/{prefix}_{timestamp}.jpg"
        img.save(filename)
        print(f"   💾 Debug image saved: {filename}", file=sys.stderr)
    except Exception as e:
        print(f"   ⚠️ Could not save debug image: {e}", file=sys.stderr)


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
        if hasattr(MODEL, 'gamma'):
            print(f"   Gamma (RBF): {MODEL.gamma}", file=sys.stderr)
        if hasattr(MODEL, 'C'):
            print(f"   C parameter: {MODEL.C}", file=sys.stderr)
        
        sys.stderr.flush()
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return False


def extract_features(image_bytes, save_debug=False):
    """Extract 1280-D features using MobileNetV2"""
    global FEATURE_EXTRACTOR
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        print(f"   📸 Original: size={img.size}, mode={img.mode}, format={img.format}", file=sys.stderr)
        
        # Save original image for debugging
        if save_debug:
            save_debug_image(img, "1_original")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"   🔄 Converted to RGB", file=sys.stderr)
        
        # Save after RGB conversion
        if save_debug:
            save_debug_image(img, "2_rgb")
        
        # Check image quality
        width, height = img.size
        if width < 224 or height < 224:
            print(f"   ⚠️ WARNING: Low resolution {width}x{height} (min 224x224)", file=sys.stderr)
        
        # Apply resize
        img_resized = img.resize((256, 256), Image.BICUBIC)
        if save_debug:
            save_debug_image(img_resized, "3_resized_256")
        
        # Apply center crop
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        img_cropped = img_resized.crop((left, top, left + 224, top + 224))
        if save_debug:
            save_debug_image(img_cropped, "4_cropped_224")
        
        # Apply full preprocessing
        img_tensor = normalize_transform(img).unsqueeze(0).to(DEVICE)
        
        print(f"   ✅ Tensor shape: {img_tensor.shape}", file=sys.stderr)
        print(f"   📊 Tensor stats: min={img_tensor.min():.3f}, max={img_tensor.max():.3f}, mean={img_tensor.mean():.3f}", file=sys.stderr)
        
        # Extract features
        with torch.no_grad():
            features = FEATURE_EXTRACTOR(img_tensor)
        
        features_np = features.cpu().numpy()
        
        print(f"   ✅ Features: shape={features_np.shape}, min={features_np.min():.3f}, max={features_np.max():.3f}, mean={features_np.mean():.3f}", file=sys.stderr)
        sys.stderr.flush()
        
        return features_np
        
    except Exception as e:
        print(f"❌ Error extracting features: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        raise


def recognize_currency_from_bytes(image_bytes, save_debug=False):
    """
    Recognize currency from image bytes using MobileNetV2 + SVM
    
    Args:
        image_bytes: Raw image bytes
        save_debug: If True, save preprocessing steps for debugging
    """
    global MODEL, FEATURE_EXTRACTOR
    
    if MODEL is None or FEATURE_EXTRACTOR is None:
        raise Exception("Models not loaded. Please wait for initialization.")
    
    try:
        print("🔍 Starting currency recognition...", file=sys.stderr)
        sys.stderr.flush()
        
        # Extract features
        features = extract_features(image_bytes, save_debug=save_debug)
        
        # Get prediction
        prediction = MODEL.predict(features)
        print(f"   🎯 Prediction: {prediction[0]}", file=sys.stderr)
        
        # Get probabilities
        try:
            probabilities = MODEL.predict_proba(features)[0]
            confidence = float(np.max(probabilities) * 100)
            
            # Print all probabilities for debugging
            currencies_short = ["10SR", "100SR", "20SR", "200SR", "5SR", "50SR", "500SR"]
            print(f"   📊 All probabilities:", file=sys.stderr)
            for i, (curr, prob) in enumerate(sorted(zip(currencies_short, probabilities), key=lambda x: x[1], reverse=True)):
                print(f"      {curr:6} : {prob*100:5.2f}%", file=sys.stderr)
            
            print(f"   📈 Max confidence: {confidence:.2f}%", file=sys.stderr)
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
                "advice": "تأكد من: 1) الإضاءة الجيدة 2) وضوح العملة 3) عدم وجود ظلال"
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
        "confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
        "debug_mode": True
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
