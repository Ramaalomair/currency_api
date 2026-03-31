import torchvision.models as models
import torch.nn as nn

print("📥 Downloading MobileNetV2 weights...")
mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
_ = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d((1, 1)))
print("✅ MobileNetV2 weights cached!")
print("🎉 Pre-download complete!")
