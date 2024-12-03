from .import corridor
from . import home
from . import market
from .continual_params import *
from .offline_params import *
import os

def get_params(continual=False, **kwargs):  
    match kwargs["sequence"]:
        case "corridor":
            config = corridor
        case "home":
            config = home
        case "market":
            config = market
        case _:
            raise ValueError("specified sequence not present in openloris configuration") 
    params = {"clustering": config.get_clustering_params(), 
             "dataset": config.get_dataset_params()}
    if continual:
        params = {**params, **get_continual_params(sequence=kwargs["sequence"])}
        params["test"]["avalanche_streams_name"] = [f'test_{s.split(os.path.sep)[-1]}' for s in params["dataset"]["test"]["sequences"]]
        return params 
    else:
        return {**params, **get_offline_params(sequence=kwargs["sequence"])}
            
__all__ = ["get_params"]