# models/__init__.py

import torch
from .convnext import CustomConvNeXt
from .resnet50 import CustomResNet50

def load_classifier_model(arch: str, num_classes: int, weights: str, device):
    if arch.lower() == "convnext":
        model = CustomConvNeXt(num_classes=num_classes)
    else:
        model = CustomResNet50(num_classes=num_classes)
    state = torch.load(weights, map_location=device,weights_only=False)
    model.load_state_dict(state)
    return model.to(device)
