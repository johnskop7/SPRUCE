# cnn_models.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import alexnet
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, convnext_tiny, ConvNeXt_Tiny_Weights


class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x, return_features=True):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        features = self.model.avgpool(x)
        features = torch.flatten(features, 1)
        logits = self.model.fc(features)

        if return_features:
            return logits, features
        else:
            return logits


class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the CustomConvNeXt model.

        Parameters:
        - num_classes (int): Number of output classes. Default is 2 for binary classification.
        - pretrained (bool): If True, loads pretrained weights. Default is True.
        """
        super(CustomConvNeXt, self).__init__()

        # Load a pretrained ConvNeXt-Base model
        if pretrained:
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
            self.convnext = convnext_base(weights=weights)
        else:
            self.convnext = convnext_base(weights=None)

        in_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)

        # Initialize the new classifier layer (optional but recommended)
        nn.init.xavier_uniform_(self.convnext.classifier[2].weight)
        if self.convnext.classifier[2].bias is not None:
            nn.init.zeros_(self.convnext.classifier[2].bias)

    def forward(self, x, return_features=True):
        """
        Forward pass that returns logits and features.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, H, W]
            return_features (bool): If True, returns (logits, features). Else, returns logits.

        Returns:
            logits (torch.Tensor): Output logits from the classifier.
            features (torch.Tensor): Features extracted before the classifier.
        """
        # Pass through the feature extractor
        x = self.convnext.features(x)
        x = self.convnext.avgpool(x)
        x = self.convnext.classifier[0](x)
        x = self.convnext.classifier[1](x)
        features = x  # Features before the classifier
        logits = self.convnext.classifier[2](x)  # Logits from the Linear layer

        if return_features:
            return logits, features
        else:
            return logits


def load_cnn_model(classifier_type, model_weights=None, num_classes=2):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and test CNN model.")
    parser.add_argument("--model_type", type=str, required=True, choices=["convnext", "resnet50"],
                        help="Type of CNN model to load.")
    parser.add_argument("--model_weights", type=str, help="Path to pretrained model weights.")
    args = parser.parse_args()

    # Load the model and print to verify
    model = load_cnn_model(args.model_type, args.model_weights)
    print(
        f"Loaded {args.model_type} model with weights from {args.model_weights if args.model_weights else 'default initialization'}")
