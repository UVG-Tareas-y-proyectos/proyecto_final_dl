import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(name, num_classes, pretrained=True):
    if num_classes < 2:
        raise ValueError("Se necesitan al menos dos clases.")
    if name == "cnn":
        blocks = []
        channels = 3
        for output in (16, 32, 64):
            blocks.extend([nn.Conv2d(channels, output, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            channels = output
        return nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, num_classes))
    if name == "mobilenet":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    raise ValueError(f"Modelo desconocido: {name}")

