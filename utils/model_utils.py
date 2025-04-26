# cnn_models.py

import torch
import torch.nn as nn
from torchvision import models
from models.ConvNeXt import CustomConvNeXt
from models.ResNet50 import CustomResNet50

def load_classifier_model(classifier_type, model_weights=None, num_classes=2):
    """Loads the specified CNN model with optional pretrained weights."""
    if classifier_type == "resnet50":
        model = CustomResNet50(num_classes)
    elif classifier_type == "convnext":
        model = CustomConvNeXt(num_classes)
    else:
        raise ValueError("Unsupported classifier_type type. Choose from  'convnext', 'resnet50'.")

    # Load custom weights if provided
    if model_weights:
        model.load_state_dict(torch.load(model_weights, weights_only=False))

    model.eval()  # Set to evaluation mode by default
    return model
