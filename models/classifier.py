import torch.nn as nn
import math
import torch
from torchvision import models


original_model = models.resnet50(pretrained=True)
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.features = nn.Sequential(
            *list(original_model.children())[:-1]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048,20),
        )

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x