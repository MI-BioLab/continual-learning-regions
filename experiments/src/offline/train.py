from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy

import os
import matplotlib

matplotlib.use('Agg')
import sys

sys.path.insert(1, "src/")
from .loop import loop
from setting_strategies import get_setting_strategy
from datasets import get_dataset
from losses import get_loss_function
from utils import (create_next_folder_in_directory,   
                   get_weighting_function,
                   get_optimizer, 
                   save_parameters_to_yaml, 
                   save_model_checkpoint,
                   write_to_tensorboard,
                   io_utils)
from models import FeatureExtractor, Classifier, Model
import globals

def train(params, device="cuda"):
    seed = params["seed"]
    # set the seed for pytorch reproducibility
    torch.manual_seed(seed)
    top_k = [1, 3, 5]
    
   
    save_folder = params["save_folder"]
    strategy_params = params["strategy"]
    train_params = params["train"]
    #clustering_type not needed
    use_validation = params["dataset"]["use_validation"]
    if use_validation:
        validation_params = params["validation"]
    strategy_params = params["strategy"]

    model_params = params["model"]    
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    new_folder = create_next_folder_in_directory(save_folder)
    
    if globals.debug:
        globals.debug_file = os.path.join(new_folder, "debug.log")
        if not params["dataset"]["load_dataset"]:
            globals.clustering_file = os.path.join(params["dataset"]["dataset_folder"], "clustering.log")
    
    strategy = get_setting_strategy(strategy_params["type"],
                                    model=None,
                                    **strategy_params)       
    
    #dataset
    dynamic_train_dataset, \
    nodes_moved_in_each_experience, \
    end_train_dataset, \
    end_valid_dataset, \
    test_dataset = get_dataset(params, strategy, continual=False)
    
    assert end_train_dataset is not None, "offline train end_train_dataset must not be None"
    assert end_valid_dataset is not None, "offline train end_valid_dataset must not be None"
        
    feature_extractor = FeatureExtractor(device=device, **model_params)

    # compute the number of classes for the training and the samples per class for the weighted loss function
    unique_classes, samples_per_class = np.unique(end_train_dataset.targets,
                                                  return_counts=True)
    num_classes = (np.max(unique_classes) + 1).item()
    samples_per_class = torch.as_tensor(samples_per_class)
    
    classifier = Classifier(feature_extractor.output_features,
                            num_classes,
                            incremental=model_params["incremental"],
                            device=device)

    model = Model(feature_extractor, classifier, device=device)
    strategy.model = model

    train_weights = None
    if "loss_weighting" in train_params:
        train_weighting_fn = get_weighting_function(
            train_params["loss_weighting"]["weighting_method"],
            **train_params["loss_weighting"])
        train_weights = train_weighting_fn(samples_per_class)

    validation_weights = None
    if use_validation:
        if "loss_weighting" in validation_params:
            _, valid_samples_per_class = np.unique(end_valid_dataset.targets,
                                                   return_counts=True)
            validation_weighting_fn = get_weighting_function(
                validation_params["loss_weighting"]["weighting_method"],
                **validation_params["loss_weighting"])
            validation_weights = validation_weighting_fn(
                valid_samples_per_class)

    train_loss_fn = get_loss_function(train_params["loss"]["loss_fn"],
                                      weight=train_weights,
                                      **train_params["loss"])
    if use_validation:
        valid_loss_fn = get_loss_function(validation_params["loss"]["loss_fn"],
                                          weight=validation_weights,
                                          **validation_params["loss"])

    num_workers = 8  # Number of data loader workers
    persistent_workers = True
    if train_params["balance_minibatches"]:
        minibatches_weighting_fn = get_weighting_function(
            train_params["minibatches_weighting"]["weighting_method"],
            **train_params["minibatches_weighting"])
        minibatches_weights = minibatches_weighting_fn(
            samples_per_class)

        #TODO check this
        samples_weights = torch.as_tensor(
            [minibatches_weights[t] for t in end_train_dataset.targets])
        sampler = WeightedRandomSampler(
            samples_weights,
            len(samples_weights),
            replacement=train_params["replacement"])
        train_dataloader = DataLoader(end_train_dataset,
                                      batch_size=train_params["batch_size"],
                                      sampler=sampler,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      persistent_workers=persistent_workers,
                                      pin_memory=True)

    else:
        train_dataloader = DataLoader(end_train_dataset,
                                      batch_size=train_params["batch_size"],
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      persistent_workers=persistent_workers,
                                      pin_memory=True)

    if use_validation:
        valid_dataloader = DataLoader(
            end_valid_dataset,
            batch_size=validation_params["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True)    

    

    lr = train_params["initial_learning_rate"]
    optimizer = get_optimizer(train_params["optimizer"]["type"],
                              model.parameters(), lr,
                              **train_params["optimizer"])

    params["model"]["classifier"] = str(model.classifier)
    params["model"]["freezed"] = str(model.feature_extractor.freezed_part)
    params["model"]["trainable"] = str(model.feature_extractor.trainable_part)
    save_parameters_to_yaml(os.path.join(new_folder, io_utils.current_params_file), params)

    min_loss = np.inf
    max_accuracy = 0
    n_epoch_since_last_best_accuracy = 0
    writer = SummaryWriter(log_dir=os.path.join(new_folder, io_utils.tensorboard_log_dir),
                           comment="train")

    # training cycle
    for epoch in range(train_params["epochs"]):
        print(f"Epoch {epoch+1}\n-------------------------------")
        if (train_params["use_learning_rate_decay"]
                and n_epoch_since_last_best_accuracy
                == train_params["patience_for_learning_rate_decay"]):
            n_epoch_since_last_best_accuracy = 0
            lr *= train_params["learning_rate_multiplier"]
            for g in optimizer.param_groups:
                g["lr"] = lr
                print(f"Learning rate decreased. New learning rate: {lr:.6f}")
            if lr < train_params["min_learning_rate"]:
                if train_params["use_early_stopping"]:
                    break
                lr = train_params["min_learning_rate"]

        train_loss, train_accuracy, train_cm = loop(
            strategy,
            train_dataloader,
            train_loss_fn,
            optimizer=optimizer,
            accuracy_fn=lambda predictions, targets, num_classes, top_k:
            accuracy(
                preds=predictions,
                target=targets,
                task="multiclass",
                num_classes=num_classes,
                top_k=top_k,
            ),
            num_classes=num_classes,
            top_k=top_k,
            train=True,
        )

        if use_validation:
            valid_loss, valid_accuracy, valid_cm = loop(
                strategy,
                valid_dataloader,
                valid_loss_fn,
                accuracy_fn=lambda predictions, targets, num_classes, top_k:
                accuracy(
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

        if min_loss > (valid_loss if use_validation else train_loss):
            if train_params["save_best_model"]:
                save_model_checkpoint(model, new_folder,
                                      io_utils.best_prefix + io_utils.loss)
            min_loss = valid_loss if use_validation else train_loss

        if max_accuracy < (valid_accuracy[0]
                           if use_validation else train_accuracy[0]):
            print(
                f"Accuracy Increased({max_accuracy:.6f}--->{(valid_accuracy[0] if use_validation else train_accuracy[0]):.6f}) \t Saving The Model"
            )
            max_accuracy = valid_accuracy[
                0] if use_validation else train_accuracy[0]
            if train_params["save_best_model"]:
                save_model_checkpoint(model, new_folder,
                                      io_utils.best_prefix + io_utils.accuracy)
            n_epoch_since_last_best_accuracy = 0
        else:
            n_epoch_since_last_best_accuracy += 1

        print("Train loss: " + str(train_loss))
        write_to_tensorboard(
            writer,
            loss_tag="Loss/training/",
            loss=train_loss,
            accuracy_tag="Accuracy/training/",
            accuracy=train_accuracy,
            confusion_matrix_tag="Confusion Matrix/training/",
            confusion_matrix=train_cm,
            top_k=top_k,
            step=epoch,
        )
        if use_validation:
            write_to_tensorboard(
                writer,
                loss_tag="Loss/validation/",
                loss=valid_loss,
                accuracy_tag="Accuracy/validation/",
                accuracy=valid_accuracy,
                confusion_matrix_tag="Confusion Matrix/validation/",
                confusion_matrix=valid_cm,
                top_k=top_k,
                step=epoch,
            )
            print("Valid loss: " + str(valid_loss))

        writer.add_scalar("Learning Rate", lr, epoch)
    print("Done!")
    writer.close()

    if train_params["save_end_model"]:
        save_model_checkpoint(model, new_folder, io_utils.end)

    return new_folder, num_classes
