from copy import copy
import dill
import torch

from avalanche.training.determinism.rng_manager import RNGManager
import globals

def maybe_load_checkpoint(fname, map_location=None):
    if globals.debug:
        with open(globals.debug_file, "a") as file:
            file.write("===================== RESTORED FROM HERE =====================\n")
            file.write(f"Loading checkpoint {fname}\n") 
    ckp = torch.load(fname, pickle_module=dill, map_location=map_location)

    print(ckp)
    strategy = ckp["strategy"]
    rng_manager_dict = ckp["rng_manager"].__dict__
    sequence_counter = ckp["sequence_counter"]
    experience_counter = ckp["experience_counter"]
    return strategy, rng_manager_dict, sequence_counter, experience_counter


def save_checkpoint(strategy, sequence_counter, experience_counter, fname, exclude=None): 
    if globals.debug:
        with open(globals.debug_file, "a") as file:
            file.write(f"Saving checkpoint {fname}\n")       
    if exclude is None:
        exclude = []

    strategy = copy(strategy)
    for attr in exclude:
        delattr(strategy, attr)

    checkpoint_data = {
        "strategy": strategy,
        "rng_manager": RNGManager,
        "sequence_counter": sequence_counter,
        "experience_counter": experience_counter
    }
    torch.save(checkpoint_data, fname, pickle_module=dill)

__all__ = ["maybe_load_checkpoint",
           "save_checkpoint"]