from .KITTI_dataset import KITTIDataset
from .OpenLoris_dataset import OpenLorisDataset
from .TUM_dataset import TUMDataset
from .StLucia_dataset import StLuciaDataset

def dataset_factory(name):
    if name == "kitti":
        dataset = KITTIDataset()
      
    elif name == "openloris":
        dataset = OpenLorisDataset()

    elif name == "tum":
        dataset = TUMDataset()
        
    elif name == "stlucia":
        dataset = StLuciaDataset()
    
    return dataset

__all__ = ["dataset_factory"]