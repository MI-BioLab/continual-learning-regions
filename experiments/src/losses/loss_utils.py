from torch import nn

from losses import FocalLoss


def get_loss_function(value, **kwargs):
    match value:
        case 0:
            return nn.CrossEntropyLoss()
        case 1:
            return FocalLoss(kwargs["gamma"])
        
    raise ValueError("get_loss_function: invalid loss function") 
        
        
__all__ = ["get_loss_function"]