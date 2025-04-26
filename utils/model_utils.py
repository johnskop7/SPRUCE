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


def load_generators(old_generator_path, new_generator_path):
    """Loads the old and new StyleGAN generators."""
    try:
        with open(old_generator_path, 'rb') as f:
            old_G = pickle.load(f)['G_ema']
        with open(new_generator_path, 'rb') as f_new:
            new_G = torch.load(f_new, map_location="cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            old_G, new_G = old_G.cuda(), new_G.cuda()
        return old_G, new_G
    except Exception as e:
        print(f"Error loading generators: {e}")
        return None, None
