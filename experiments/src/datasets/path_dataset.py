from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize

class PathDataset(Dataset):
 
    def __init__(self, 
                 ids: Sequence,
                 paths: Sequence, 
                 targets: Sequence, 
                 height: int=224, 
                 width: int=224):
        
        self.ids = np.array(ids).astype(np.int64)
        self.paths = np.array(paths).astype(np.string_)
        self.targets = np.array(targets).astype(np.int64)
        self.height = height
        self.width = width
        self.resize = Resize((height, width))
                        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        images = []
        for path in self.paths[idx]:
            image = self.resize(read_image(path).float() / 255)
            images.append(image)
        
        return self.ids[idx], images, torch.tensor(self.targets[idx])
    
__all__ = ["PathDataset"]