from typing import Sequence

import yaml
yaml.Dumper.ignore_aliases = lambda *args : True

import numpy as np
import pandas as pd
from glob import glob
import os
import torch
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter


pytorch_ext = ".pt"
json_ext = ".json"
loss = "loss"
accuracy = "accuracy"
tensorboard_log_dir = "tb_data/"
current_params_file = "current_params.yaml"
new_folder_prefix = "experiment_"
serialized_postfix = "_serialized"
best_prefix = "best_"
end = "end"
dynamic_train_dataset_file = "dynamic_train_dataset" + json_ext
end_train_dataset_file = "end_train_dataset" + json_ext
end_valid_dataset_file = "end_valid_dataset" + json_ext
test_dataset_file = "test_dataset" + json_ext
checkpoint_file = "checkpoint.pkl"

def read_txt(filename: str, separator: str=" " , header=None):
    return pd.read_csv(filename, sep=separator, header=header).to_numpy()   

def merge_paths(root, sequence):
    return [os.path.join(root, s) if root is not None else s for s in sequence]
    
def create_next_folder_in_directory(saving_root, prefix=new_folder_prefix):
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    folders_in_saving_root = glob(os.path.join(saving_root, "*"))
    current_folder_number = str(len(folders_in_saving_root))
    new_folder = os.path.join(saving_root, prefix + current_folder_number)
    os.mkdir(new_folder)
    return new_folder

def get_parameters_from_yaml(file):
    with open(file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
def save_parameters_to_yaml(file, parameters):
    with open(file, "w") as stream:
        try:
            yaml.dump(parameters, stream)
        except yaml.YAMLError as exc:
            print(exc)
            

def write_to_tensorboard(
    writer: SummaryWriter,
    loss_tag: str,
    loss: float,
    accuracy_tag: str,
    accuracy: Sequence[float],
    confusion_matrix_tag: str = None,
    confusion_matrix: np.ndarray = None,
    top_k: Sequence[int] = [1, 3, 5],
    step: int = None,
):
    writer.add_scalar(loss_tag, loss, step)
    for i in range(len(top_k)):
        writer.add_scalar(accuracy_tag + "top-" + str(top_k[i]), accuracy[i], step)
    if confusion_matrix is not None and confusion_matrix_tag is not None:
        test_cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=np.arange(len(confusion_matrix)),
        )
        test_cm_display = test_cm_display.plot(include_values=False)
        writer.add_figure(confusion_matrix_tag, test_cm_display.figure_, step)


def save_model_checkpoint(
    model: torch.nn.Module,
    root_folder: str,
    model_name: str,
    serialized_postfix: str = serialized_postfix,
    ext: str = pytorch_ext,
):
    torch.save(model.state_dict(), os.path.join(root_folder, model_name + ext))
    model_serialized = torch.jit.script(model)
    model_serialized.save(os.path.join(root_folder, model_name + serialized_postfix + ext))


__all__ = [
    "read_txt",
    "merge_paths",
    "create_next_folder_in_directory",
    "get_parameters_from_yaml",
    "save_parameters_to_yaml",
    "write_to_tensorboard",
    "save_model_checkpoint",
    "pytorch_ext",
    "json_ext",
    "loss",
    "accuracy",
    "tensorboard_log_dir",
    "current_params_file",
    "new_folder_prefix",
    "best_prefix",
    "end",
    "dynamic_train_dataset_file",
    "end_train_dataset_file",
    "end_valid_dataset_file",
    "test_dataset_file",
    "checkpoint_file"
]
