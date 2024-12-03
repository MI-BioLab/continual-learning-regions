import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.functional import accuracy

import matplotlib

matplotlib.use("Agg")
import sys

sys.path.insert(1, "src/")
from .loop import loop
from setting_strategies import get_setting_strategy
from datasets import get_dataset
from losses import get_loss_function
from utils import (
    get_weighting_function,
    write_to_tensorboard,
    io_utils
)
from setting_strategies import BaseStrategy

def test_model(writer,
               strategy: BaseStrategy,
               sequence_name: str,
               model_name: str,
               test_dataloader: DataLoader,
               test_loss_fn: torch.nn.Module,
               num_classes: int,
               top_k = [1, 3, 5],
               ):
    
    strategy.model.eval()

    test_loss, test_accuracy, test_cm = loop(
        strategy,
        test_dataloader,
        loss_fn=test_loss_fn,
        accuracy_fn=lambda predictions, targets, num_classes, top_k: accuracy(
            preds=predictions,
            target=targets,
            task="multiclass",
            num_classes=num_classes,
            top_k=top_k,
        ),
        num_classes=num_classes,
        top_k=top_k,
        train=False,
    )

    write_to_tensorboard(
        writer,
        loss_tag="Loss/test/" + os.path.join(sequence_name, model_name) + "/",
        loss=test_loss,
        accuracy_tag="Accuracy/test/" + os.path.join(sequence_name, model_name) + "/",
        accuracy=test_accuracy,
        confusion_matrix_tag="Confusion Matrix/test/" + os.path.join(sequence_name, model_name) + "/",
        confusion_matrix=test_cm,
        top_k=top_k,
    )

def test(params, train_folder, num_classes, device="cuda"):
    
    test_params = params["test"]

    batch_size = test_params["batch_size"]
    top_k = [1, 3, 5]

    strategy_params = params["strategy"]
    
    strategy = get_setting_strategy(strategy_params["type"], model=None, **strategy_params)
    _, _, _, _, test_dataset = get_dataset(params, strategy, continual=False)   

    assert test_dataset is not None, "offline test test_dataset must not be None"
   
    writer = SummaryWriter(
        log_dir=os.path.join(train_folder, io_utils.tensorboard_log_dir), comment="test"
    )

    sequences = params["dataset"]["test"]["sequences"]
    for i in range(len(sequences)):
        unique_classes, samples_per_class = np.unique(test_dataset[i][0].targets, return_counts=True)
        samples_per_class = torch.as_tensor(samples_per_class)

        test_weights = None
        if "loss_weighting" in test_params:
            test_weighting_fn = get_weighting_function(test_params["loss_weighting"]["weighting_method"],
                                                      **test_params["loss_weighting"])
            test_weights = test_weighting_fn(samples_per_class)
            
        test_loss_fn = get_loss_function(test_params["loss"]["loss_fn"],
                                        weights=test_weights, 
                                        **test_params["loss"])

        test_dataloader = DataLoader(
            test_dataset[i][0], batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, persistent_workers=False, pin_memory=True
        )

        models_to_test = [io_utils.best_prefix + io_utils.loss, io_utils.best_prefix + io_utils.accuracy, io_utils.end]
        _, sequence_name = os.path.split(sequences[i])
        # print(sequence_name)
        for name in models_to_test:
            strategy.model = torch.jit.load(os.path.join(train_folder, name
                                            + io_utils.serialized_postfix 
                                            + io_utils.pytorch_ext
                                            ))
            
            test_model(writer, strategy, sequence_name, name, test_dataloader, test_loss_fn, num_classes, top_k)
    writer.close()


if __name__ == "__main__":
    config_dir = "./config/usyd_dataset"
    config_file = "offline.yaml"
    # params = parse_params(config_dir, config_file)
    # test(params, "./res/offline/" + io_utils.new_folder_prefix + "0/", 18)
