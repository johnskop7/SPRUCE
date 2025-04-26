import torch
import torch.nn as nn
from torchvision import models

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
