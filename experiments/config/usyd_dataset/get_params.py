from .clustering_params import *
from .dataset_params import *
from .continual_params import *
from .offline_params import *

import os

def get_params(continual=False, **kwargs):   
    params = {"clustering": get_clustering_params(), 
             "dataset": get_dataset_params()}
    if continual:
        params = {**params, **get_continual_params(sequence=kwargs["sequence"])}
        params["test"]["avalanche_streams_name"] = [f'test_{s.split(os.path.sep)[-1].split("_")[0]}' for s in params["dataset"]["test"]["sequences"]]
        return params
    else:
        return {**params, **get_offline_params(sequence=kwargs["sequence"])}
            
__all__ = ["get_params"]