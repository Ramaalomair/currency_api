import torchvision.models as models
import torch.nn as nn
from rembg import new_session

print("📥 Downloading MobileNetV2 weights...")
mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
_ = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d((1, 1)))
print("✅ MobileNetV2 weights cached!")

print("📥 Downloading rembg u2net model...")
session = new_session("u2net")
print("✅ rembg u2net model cached!")

print("🎉 All models pre-downloaded successfully!")
