import os

import sys
sys.path.insert(1, "src/")
from utils import get_parameters_from_yaml, parse_params_for_training

def parse_params(config_dir, config_file):
    parameters = get_parameters_from_yaml(os.path.join(config_dir, config_file))
   
    parse_params_for_training(config_dir, parameters)
    
    parameters["model"]["incremental"] = True
    
    return parameters
    
if __name__ == "__main__":
    params = parse_params("./config/usyd_dataset", "continual.yaml")
    print(params)
    