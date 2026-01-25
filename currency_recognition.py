import joblib
import numpy as np
from PIL import Image
import io
import os
import urllib.request
import cv2
import sys

# متغيرات عامة
MODEL = None
MODEL_PATH = "SVM_(RBF).pkl"

# رابط GitHub Release
MODEL_URL = "https://github.com/Ramaalomair/currency_api/raw/main/models/currency/SVM_%28RBF%29.pkl"

def initialize_currency_recognition():
    """تحميل موديل SVM مرة واحدة"""
    global MODEL
    
    print("=" * 60, file=sys.stderr)
    print("🔄 INITIALIZING CURRENCY RECOGNITION MODEL", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.stderr.flush()
    
    if MODEL is not None:
        print("✅ Model already loaded!", file=sys.stderr)
        sys.stderr.flush()
        return True
    
    try:
        # تحميل الموديل من GitHub Releases إذا مو موجود
        if not os.path.exists(MODEL_PATH):
            print(f"📥 Downloading SVM model from GitHub Releases...", file=sys.stderr)
            print(f"   URL: {MODEL_URL}", file=sys.stderr)
            sys.stderr.flush()
            
            # تحميل الملف
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            file_size = os.path.getsize(MODEL_PATH)
            print(f"✅ Model downloaded successfully! ({file_size} bytes)", file=sys.stderr)
            sys.stderr.flush()
        else:
            file_size = os.path.getsize(MODEL_PATH)
            print(f"✅ Model file already exists locally ({file_size} bytes)", file=sys.stderr)
            sys.stderr.flush()
        
        # تحميل موديل SVM في الذاكرة
        print("🔄 Loading SVM model into memory...", file=sys.stderr)
        sys.stderr.flush()
        MODEL = joblib.load(MODEL_PATH)
        print("✅ SVM Model loaded and ready!", file=sys.stderr)
        print(f"   Model type: {type(MODEL)}", file=sys.stderr)
        
        # معلومات إضافية عن الموديل
        if hasattr(MODEL, 'classes_'):
            print(f"   Classes: {MODEL.classes_}", file=sys.stderr)
        if hasattr(MODEL, 'n_support_'):
            print(f"   Support vectors: {MODEL.n_support_}", file=sys.stderr)
        
        sys.stderr.flush()
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return False

def preprocess_image(image_bytes, target_size=(128, 128)):
    """معالجة الصورة قبل التنبؤ"""
    try:
        # فتح الصورة
        img = Image.open(io.BytesIO(image_bytes))
        
        # تحويل لـ RGB إذا كانت RGBA
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # تحويل لـ numpy array
        img_array = np.array(img)
        
        # تحويل من RGB لـ BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # تغيير الحجم
        img_resized = cv2.resize(img_bgr, target_size)
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        # تحويل لـ feature vector (flatten)
        features = img_normalized.flatten().reshape(1, -1)
        
        return features
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        raise

def recognize_currency_from_bytes(image_bytes):
    """التعرف على العملة من bytes الصورة باستخدام SVM"""
    global MODEL
    
    if MODEL is None:
        raise Exception("Model not loaded. Please wait for initialization.")
    
    try:
        print("🔍 Starting currency recognition...", file=sys.stderr)
        sys.stderr.flush()
        
        # معالجة الصورة
        features = preprocess_image(image_bytes)
        print(f"   Features shape: {features.shape}", file=sys.stderr)
        sys.stderr.flush()
        
        # التنبؤ
        prediction = MODEL.predict(features)
        print(f"   Prediction: {prediction}", file=sys.stderr)
        sys.stderr.flush()
        
        # الحصول على احتمالات التنبؤ (إذا كان الموديل يدعمها)
        try:
            probabilities = MODEL.predict_proba(features)
            confidence = float(np.max(probabilities) * 100)
            print(f"   Probabilities: {probabilities}", file=sys.stderr)
            print(f"   Confidence: {confidence:.2f}%", file=sys.stderr)
            sys.stderr.flush()
        except AttributeError:
            # إذا الموديل ما يدعم predict_proba
            confidence = 100.0
            print("   (Model doesn't support probability prediction - using 100%)", file=sys.stderr)
            sys.stderr.flush()
        
        # قائمة العملات - عدّلها حسب موديلك
        # مهم: الترتيب لازم يكون نفس ترتيب الـ labels اللي دربت عليها الموديل
        currencies = {
            0: "10 SR",
            1: "50 SR",
            2: "100 SR",
            3: "500 SR"
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
    """الحصول على حالة الموديل"""
    status = {
        "initialized": MODEL is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_url": MODEL_URL
    }
    
    if MODEL is not None:
        status["model_type"] = str(type(MODEL))
        
        # معلومات إضافية عن الموديل
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
    """للتوافق مع الكود القديم"""
    return MODEL
