import torch.nn as nn
from torchvision import models

def get_resnet50_model(num_classes=5):
    model = models.resnet50(weights='DEFAULT')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
