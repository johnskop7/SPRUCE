import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class CustomEfficientNetB3(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        EfficientNet-B3 fine-tuner that can return logits and penultimate features.

        Args:
          num_classes (int): number of output classes (default=2).
          pretrained (bool): load ImageNet-pretrained weights (default=True).
        """
        super().__init__()
        # 1) Load base model
        if pretrained:
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)

        # 2) Replace classifier head
        #    original classifier is: [Dropout, Linear(in_features, 1000)]
        in_features = self.backbone.classifier[1].in_features
        # new head: keep dropout + linear to num_classes
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # 3) (Optional) weight init for new layer
        nn.init.xavier_uniform_(self.backbone.classifier[1].weight)
        if self.backbone.classifier[1].bias is not None:
            nn.init.zeros_(self.backbone.classifier[1].bias)

    def forward(self, x, return_features=True):
        """
        Args:
          x (Tensor): [B, 3, H, W]
          return_features (bool): if True, also return the penultimate features.

        Returns:
          logits: [B, num_classes]
          features (optional): [B, in_features]
        """
        # 1) Stem + feature blocks
        x = self.backbone.features(x)

        # 2) Global pool
        x = self.backbone.avgpool(x)               # [B, C, 1, 1]
        x = torch.flatten(x, 1)                    # [B, C]

        # 3) Dropout (same as original)
        features = self.backbone.classifier[0](x)  # [B, C] after dropout

        # 4) Final linear
        logits = self.backbone.classifier[1](features)  # [B, num_classes]

        if return_features:
            return logits, features
        return logits
