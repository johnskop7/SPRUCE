import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, convnext_tiny, ConvNeXt_Tiny_Weights


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
