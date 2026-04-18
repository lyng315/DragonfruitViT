import torch.nn as nn
from torchvision import models

def get_vgg16_model(num_classes=5):
    model = models.vgg16(weights='DEFAULT')
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model
