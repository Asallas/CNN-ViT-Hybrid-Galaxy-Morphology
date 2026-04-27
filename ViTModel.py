import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
def get_vit_model():
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        
        model = vit_b_16(weights=weights)
        
        model.heads.head = nn.Linear(model.heads.head.in_features, 3)

        return model