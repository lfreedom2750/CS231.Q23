import torch.nn as nn
from torchvision import models


def build_vgg16(num_classes: int):
    model = models.vgg16(weights=None)  # không pretrained
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model