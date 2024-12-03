import torch

def get_optimizer(value, params, lr, **kwargs):
    match value:
        case 0:
            return torch.optim.SGD(params, lr=lr, momentum=kwargs["momentum"] if "momentum" in kwargs else 0)
        case 1:
            return torch.optim.Adam(params, lr=lr)
    raise ValueError("get_optimizer: invalid optimizer")    
        
__all__ = ["get_optimizer"]