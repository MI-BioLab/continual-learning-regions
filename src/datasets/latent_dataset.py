from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

"""
Dataset composed of (features extracted at latent replay layer, label).
This dataset is used to make the predictions, passing the features through the last trainable part of the network.
"""
class LatentDataset(Dataset):
 
    def __init__(self, 
                 ids: Sequence,
                 features: torch.Tensor, 
                 targets: torch.Tensor):
        
        self.ids = np.array(ids).astype(np.string_)
        self.features = features
        self.targets = targets
                        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):                
        return self.ids[idx], self.features[idx], self.targets[idx]
    
__all__ = ["LatentDataset"]