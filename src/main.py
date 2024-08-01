import torch

import offline
import continual

# from train_dynamic import train as train_dynamic
# from continual.params import parse_params
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.backends.cudnn.benchmark = True

import sys
sys.path.insert(0, "")
from config import get_params

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    train_continual = True #TODO change everytime
    dataset = "usyd" #TODO change everytime
    sequence = None #TODO change everytime

    params = get_params(dataset, continual=train_continual, sequence=sequence)
    
    print(params)
    if train_continual:
        continual.train(params, device)
    else:
        new_folder, num_classes = offline.train(params, device)
        offline.test(params, new_folder, num_classes, device)
        
    