"""
models.py

Wrapper sencillo para instanciar EfficientNet-B0 o MobileNetV3-Large con la última capa 
adaptada al número de clases (2 para binario, 5 para multiclase)
Usa pesos preentrenados en ImageNet como punto de partida (transfer learning)
"""

import torch.nn as nn
from torchvision import models


def build_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Arquitectura no soportada: {architecture}")
