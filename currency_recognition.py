"""
currency_recognition.py - Munir Currency Recognition Module
============================================================
Saudi Riyal Banknote Recognition using:
- MobileNetV2 (pre-trained) for feature extraction
- SVM (RBF kernel) for classification

Supports: 5, 10, 20, 50, 100, 200, 500 SAR
Languages: Arabic + English
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle
import logging
from pathlib import Path
import io

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# Currency classes (7 Saudi Riyal denominations)
# Order MUST match the training order (alphabetically sorted)
CURRENCY_CLASSES = {
    0: '10 SR',
    1: '100 SR',
    2: '20 SR',
    3: '200 SR',
    4: '5 SR',
    5: '50 SR',
    6: '500 SR'
}

# Voice guidance text (Arabic + English)
CURRENCY_TEXT = {
    'arabic': {
        '5 SR': 'هذه ورقة نقدية من فئة خمسة ريال سعودي',
        '10 SR': 'هذه ورقة نقدية من فئة عشرة ريال سعودي',
        '20 SR': 'هذه ورقة نقدية من فئة عشرين ريال سعودي',
        '50 SR': 'هذه ورقة نقدية من فئة خمسين ريال سعودي',
        '100 SR': 'هذه ورقة نقدية من فئة مئة ريال سعودي',
        '200 SR': 'هذه ورقة نقدية من فئة مئتين ريال سعودي',
        '500 SR': 'هذه ورقة نقدية من فئة خمسمئة ريال سعودي'
    },
    'english': {
        '5 SR': 'This is a five Saudi Riyal banknote',
        '10 SR': 'This is a ten Saudi Riyal banknote',
        '20 SR': 'This is a twenty Saudi Riyal banknote',
        '50 SR': 'This is a fifty Saudi Riyal banknote',
        '100 SR': 'This is a one hundred Saudi Riyal banknote',
        '200 SR': 'This is a two hundred Saudi Riyal banknote',
        '500 SR': 'This is a five hundred Saudi Riyal banknote'
    }
}

# Model paths
MODEL_DIR = Path("models/currency")
SVM_MODEL_PATH = MODEL_DIR / "SVM_(RBF).pkl"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Image Preprocessing Pipeline
# ============================================================

# Same preprocessing used during training
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# Model Loading
# ============================================================

class CurrencyRecognizer:
    """Currency Recognition Model Handler"""
    
    def __init__(self):
        self.feature_extractor = None
        self.svm_model = None
        self.is_loaded = False
        
    def load_models(self):
        """Load MobileNetV2 and SVM models"""
        try:
            logger.info("="*60)
            logger.info("📥 Loading Currency Recognition Models...")
            logger.info("="*60)
            
            # 1. Load MobileNetV2 for feature extraction
            logger.info("⏳ Loading MobileNetV2...")
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.feature_extractor = mobilenet.features
            self.feature_extractor = nn.Sequential(
                self.feature_extractor,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_extractor = self.feature_extractor.to(DEVICE)
            self.feature_extractor.eval()
            logger.info("✅ MobileNetV2 loaded successfully!")
            logger.info(f"   - Feature dimension: 1280")
            logger.info(f"   - Device: {DEVICE}")
            
            # 2. Load SVM classifier
            logger.info("⏳ Loading SVM classifier...")
            
            if not SVM_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"SVM model not found at: {SVM_MODEL_PATH}\n"
                    f"Please place 'SVM_(RBF).pkl' in {MODEL_DIR}/"
                )
            
            with open(SVM_MODEL_PATH, 'rb') as f:
                self.svm_model = pickle.load(f)
            
            logger.info("✅ SVM model loaded successfully!")
            logger.info(f"   - Kernel: {self.svm_model.kernel}")
            logger.info(f"   - C parameter: {self.svm_model.C}")
            logger.info(f"   - Classes: {len(self.svm_model.classes_)}")
            
            self.is_loaded = True
            logger.info("="*60)
            logger.info("✅ Currency Recognition Models Ready!")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load currency models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract 1280-dimensional features from image using MobileNetV2
        
        Args:
            image: PIL Image object
            
        Returns:
            numpy array of shape (1280,)
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess
            img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
                features = features.view(features.size(0), -1)  # Flatten
                features = features.cpu().numpy()
            
            return features[0]  # Return 1D array (1280,)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise
    
    def recognize(self, image: Image.Image) -> dict:
        """
        Recognize Saudi currency denomination from image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Recognition result with Arabic + English text
        """
        try:
            if not self.is_loaded:
                return {
                    'success': False,
                    'error': 'Models not loaded',
                    'message': 'Currency recognition service not ready'
                }
            
            # Extract features
            features = self.extract_features(image)
            
            # Reshape for SVM (expects 2D array)
            features = features.reshape(1, -1)
            
            # Predict
            prediction = self.svm_model.predict(features)[0]
            
            # Get decision function scores for confidence
            decision_scores = self.svm_model.decision_function(features)[0]
            
            # Calculate confidence using softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / exp_scores.sum()
            confidence = float(probabilities[prediction])
            
            # Get currency denomination
            currency = CURRENCY_CLASSES.get(prediction, 'Unknown')
            
            # Extract value (e.g., '100' from '100 SR')
            currency_value = currency.split()[0] if currency != 'Unknown' else '0'
            
            # Get text in both languages
            arabic_text = CURRENCY_TEXT['arabic'].get(currency, 'عملة غير معروفة')
            english_text = CURRENCY_TEXT['english'].get(currency, 'Unknown currency')
            
            logger.info(f"✅ Recognized: {currency} (confidence: {confidence:.2%})")
            
            return {
                'success': True,
                'currency': currency,
                'currency_value': currency_value,
                'currency_unit': 'SAR',
                'confidence': round(confidence, 4),
                'confidence_percent': round(confidence * 100, 2),
                'text': {
                    'arabic': arabic_text,
                    'english': english_text
                },
                'class_index': int(prediction),
                'message': f'Recognized {currency_value} SAR',
                'message_ar': f'تم التعرف على ورقة نقدية من فئة {currency_value} ريال سعودي',
                'all_probabilities': {
                    CURRENCY_CLASSES[i]: round(float(prob), 4)
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'message': 'Currency recognition failed',
                'message_ar': 'فشل التعرف على العملة'
            }
    
    def read_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Convert image bytes to PIL Image"""
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')

# ============================================================
# Global Instance
# ============================================================

# Create global instance (will be initialized in main.py)
currency_recognizer = CurrencyRecognizer()

# ============================================================
# Helper Functions
# ============================================================

def initialize_currency_recognition():
    """Initialize currency recognition models"""
    return currency_recognizer.load_models()

def recognize_currency_from_bytes(image_bytes: bytes) -> dict:
    """
    Recognize currency from image bytes
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Recognition result dictionary (with Arabic + English)
    """
    try:
        image = currency_recognizer.read_image_from_bytes(image_bytes)
        return currency_recognizer.recognize(image)
    except Exception as e:
        logger.error(f"Error processing image bytes: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Image processing error',
            'message_ar': 'خطأ في معالجة الصورة'
        }

def recognize_currency_from_image(image: Image.Image) -> dict:
    """
    Recognize currency from PIL Image
    
    Args:
        image: PIL Image object
        
    Returns:
        Recognition result dictionary (with Arabic + English)
    """
    return currency_recognizer.recognize(image)

# ============================================================
# Status Check
# ============================================================

def get_currency_recognition_status() -> dict:
    """Get status of currency recognition service"""
    return {
        'loaded': currency_recognizer.is_loaded,
        'device': str(DEVICE),
        'num_classes': len(CURRENCY_CLASSES),
        'classes': list(CURRENCY_CLASSES.values()),
        'languages': ['arabic', 'english'],
        'model_path': str(SVM_MODEL_PATH),
        'model_exists': SVM_MODEL_PATH.exists()
    }
