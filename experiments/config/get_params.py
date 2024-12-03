from . import usyd_dataset
from . import stlucia_dataset
from . import openloris_dataset

def get_params(dataset, continual=False, **kwargs):
    match dataset:
        case "stlucia":
            return stlucia_dataset.get_params(continual, **kwargs)
        case "usyd":
            return usyd_dataset.get_params(continual, **kwargs)
        case "openloris":
            return openloris_dataset.get_params(continual, **kwargs)
            
__all__ = ["get_params"]