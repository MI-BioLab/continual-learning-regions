import torch
from torch import nn

from avalanche.models.dynamic_modules import IncrementalClassifier

class Classifier(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int,
                 incremental: bool = False,
                 device: str = "cuda",
                 **kwargs):
        super(Classifier, self).__init__()

        self.last_layer = IncrementalClassifier(input_dim, num_classes).to(device, non_blocking=True) if incremental else nn.Linear(input_dim, num_classes, device=device)
        
    def forward(self, x):
        x = self.last_layer(x)
        return torch.sigmoid(x)
