import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown
import os
from pathlib import Path

# متغيرات عامة
MODEL = None
MODEL_PATH = "currency_model.tflite"

def initialize_currency_recognition():
    """تحميل الموديل مرة واحدة"""
    global MODEL
    
    if MODEL is not None:
        return True
    
    try:
        # تحميل من Google Drive إذا مو موجود
        if not os.path.exists(MODEL_PATH):
            print("📥 Downloading model from Google Drive...")
            # ضع رابط Google Drive حقك هنا
            file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                MODEL_PATH,
                quiet=False
            )
        
        # تحميل الموديل
        print("🔄 Loading TFLite model...")
        MODEL = tf.lite.Interpreter(model_path=MODEL_PATH)
        MODEL.allocate_tensors()
        print("✅ Model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def recognize_currency_from_bytes(image_bytes):
    """التعرف على العملة"""
    global MODEL
    
    if MODEL is None:
        raise Exception("Model not loaded")
    
    # معالجة الصورة
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  # عدّل الحجم حسب موديلك
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # التنبؤ
    input_details = MODEL.get_input_details()
    output_details = MODEL.get_output_details()
    
    MODEL.set_tensor(input_details[0]['index'], img_array)
    MODEL.invoke()
    output = MODEL.get_tensor(output_details[0]['index'])
    
    # معالجة النتيجة
    confidence = float(np.max(output))
    currency_idx = int(np.argmax(output))
    
    # قائمة العملات (عدّلها حسب موديلك)
    currencies = ["10 SR", "50 SR", "100 SR", "500 SR"]
    
    return {
        "currency": currencies[currency_idx],
        "confidence": confidence * 100
    }

def get_currency_recognition_status():
    """حالة الموديل"""
    return {
        "initialized": MODEL is not None,
        "model_path": MODEL_PATH
    }

def currency_recognizer():
    """للتوافق مع الكود القديم"""
    return MODEL
