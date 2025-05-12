# models/__init__.py

import torch
from .convnext import CustomConvNeXt
from .resnet50 import CustomResNet50
from models.efficientnet import CustomEfficientNetB3

def load_classifier_model(arch: str, num_classes: int, weights: str, device):
    if arch.lower() == 'convnext':
        model = CustomConvNeXt(num_classes=num_classes)
    elif arch.lower() == 'resnet50':
        model = CustomResNet50(num_classes=num_classes)
    elif arch.lower() == 'efficientnet':
        model = CustomEfficientNetB3(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")
    state = torch.load(weights, map_location=device,weights_only=False)
    model.load_state_dict(state)
    return model.to(device)
