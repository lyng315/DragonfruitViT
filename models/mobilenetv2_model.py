import torch.nn as nn
from torchvision import models

def get_mobilenetv2_model(num_classes=5):
    model = models.mobilenet_v2(weights='DEFAULT')
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
