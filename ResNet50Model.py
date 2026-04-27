import torch.nn as nn
from torchvision.models import resnet50

def get_model():
    model = resnet50(weights="IMAGENET1K_V2")

    model.fc = nn.Linear(model.fc.in_features, 3)
    
    return model