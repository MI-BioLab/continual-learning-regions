import os

import torch
import numpy as np

from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.benchmarks import dataset_benchmark

from .continual_strategy_full_test_set import ContinualStrategy
from models import FeatureExtractor, Classifier, Model
from datasets import get_dataset
from losses import get_loss_function
from plugins import get_plugins
from setting_strategies import get_setting_strategy
from utils import (create_next_folder_in_directory, get_optimizer,
                   get_weighting_function, maybe_load_checkpoint,
                   save_checkpoint, save_parameters_to_yaml, io_utils)
import globals
import sys


def train(params, device="cuda"):
    restore = params["restore_from_checkpoint"]
    seed = params["seed"]

    save_folder = params["save_folder"]
    strategy_params = params["strategy"]
    train_params = params["train"]
    test_params = params["test"]

    model_params = params["model"]

    if not restore:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        new_folder = create_next_folder_in_directory(save_folder)
    else:
        new_folder = save_folder
    globals.new_folder = new_folder

    if globals.debug:
        globals.debug_file = os.path.join(new_folder, "debug.log")
        if not params["dataset"]["load_dataset"]:
            globals.clustering_file = os.path.join(
                params["dataset"]["dataset_folder"],
                "clustering.log")

        globals.memory_profiler_file = open(
            os.path.join(new_folder, "memory_profiler.log"), "w+")
        sys.stdout = globals.memory_profiler_file

    num_workers = 8
    pin_memory = True
    persistent_workers = False

    sequence_counter = 0
    experience_counter = 0
    if restore:
        strategy, rng_manager_dict, sequence_counter, experience_counter = maybe_load_checkpoint(
            os.path.join(params["save_folder"], io_utils.checkpoint_file))
        setting_strategy = strategy.setting_strategy
        RNGManager.__dict__.update(rng_manager_dict)
        
    else:
        RNGManager.set_random_seeds(seed)

        feature_extractor = FeatureExtractor(device=device, **model_params)

        # initially with 0 classes
        classifier = Classifier(feature_extractor.output_features,
                                0,
                                incremental=model_params["incremental"],
                                device=device)
        model = Model(feature_extractor, classifier, device=device)

        setting_strategy = get_setting_strategy(strategy_params["type"],
                                                model=model,
                                                **strategy_params)
        
    #dataset
    dynamic_train_dataset, \
    nodes_moved_in_each_experience, \
    end_train_dataset, \
    end_valid_dataset, \
    test_dataset = get_dataset(params, setting_strategy, continual=True)

    assert dynamic_train_dataset is not None, "continual train dynamic_train_dataset must not be None"
    assert nodes_moved_in_each_experience is not None, "continual train nodes_moved_in_each_experience must not be None"
    assert test_dataset is not None, "continual train test_dataset must not be None"

    unique_classes, _ = np.unique(end_train_dataset.targets,
                                return_counts=True)
    num_classes = (np.max(unique_classes) + 1).item()
    
    if not restore:
        lr = train_params["initial_learning_rate"]
        optimizer = get_optimizer(train_params["optimizer"]["type"],
                                  model.parameters(), lr,
                                  **train_params["optimizer"])
        params["model"]["classifier"] = str(model.classifier)
        params["model"]["freezed"] = str(model.feature_extractor.freezed_part)
        params["model"]["trainable"] = str(
            model.feature_extractor.trainable_part)
        save_parameters_to_yaml(
            os.path.join(new_folder, io_utils.current_params_file), params)

        #plugins
        train_weighting_fn = None
        if "loss_weighting" in train_params:
            train_weighting_fn = get_weighting_function(
                train_params["loss_weighting"]["weighting_method"],
                **train_params["loss_weighting"])

        eval_weighting_fn = None
        if "loss_weighting" in test_params:
            eval_weighting_fn = get_weighting_function(
                test_params["loss_weighting"]["weighting_method"],
                **test_params["loss_weighting"])

        train_plugins, eval_plugin = get_plugins(
            params,
            num_classes=num_classes,
            new_folder=new_folder,
            compute_train_weights_fn=train_weighting_fn,
            compute_eval_weights_fn=eval_weighting_fn)

        train_loss_fn = get_loss_function(train_params["loss"]["loss_fn"],
                                          **train_params["loss"])

        eval_loss_fn = None
        eval_loss_fn = get_loss_function(test_params["loss"]["loss_fn"],
                                         **test_params["loss"])
        strategy = ContinualStrategy(setting_strategy,
                                     nodes_moved_in_each_experience,
                                     [optimizer],
                                     [train_loss_fn, eval_loss_fn],
                                     train_mb_size=train_params["batch_size"],
                                     train_epochs=train_params["epochs"],
                                     eval_mb_size=test_params["batch_size"],
                                     plugins=train_plugins,
                                     evaluator=eval_plugin,
                                     device=device)

    

    train_benchmarks = [dataset_benchmark(t, t) for t in dynamic_train_dataset]

    test_streams = dict()
    for i, t in enumerate(test_dataset):
        test_streams[test_params["avalanche_streams_name"][i]] = t
       

    test_benchmark = dataset_benchmark(test_dataset[0],
                                       test_dataset[0],
                                       other_streams_datasets=test_streams)

    if globals.debug:
        with open(globals.debug_file, "a") as file:
            file.write(f"Num classes: {num_classes}\n")
            file.write(f"Num workers: {num_workers}\n")
            file.write(f"Pin memory: {pin_memory}\n")
            file.write(f"Persistent workers: {persistent_workers}\n")

    for i in range(sequence_counter, len(dynamic_train_dataset)):
        train_benchmark = train_benchmarks[i]

        for j in range(experience_counter, len(train_benchmark.train_stream)):
            if globals.debug:
                with open(globals.debug_file, "a") as file:
                    file.write(
                        f"Starting training on experience {j} of sequence {i}\n"
                    )
            strategy.train(train_benchmark.train_stream[j],
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           persistent_workers=persistent_workers)
            
            if test_params["test_every"] == 1:  #each experience
                test(strategy, test_streams, test_benchmark, num_workers,
                     pin_memory, persistent_workers)

            if j + 1 >= len(train_benchmark.train_stream):
                sequence_counter = i + 1
                experience_counter = 0
            else:
                experience_counter = j + 1
            save_checkpoint(strategy, sequence_counter, experience_counter,
                            os.path.join(new_folder, io_utils.checkpoint_file))

        if test_params["test_every"] == 0:  #each sequence
            test(strategy, test_streams, test_benchmark, num_workers,
                 pin_memory, persistent_workers)

    if test_params["test_every"] == -1:  #only at the end
        test(strategy, test_streams, test_benchmark, num_workers, pin_memory,
             persistent_workers)

    torch.save(setting_strategy.model.state_dict(),
               os.path.join(new_folder, "model_end.pt"))


def test(strategy, test_streams, test_benchmark, num_workers, pin_memory,
         persistent_workers):
    for name, _ in test_streams.items():
        name_stream = name + "_stream"
        if globals.debug:
            with open(globals.debug_file, "a") as file:
                file.write(f"Starting evaluation on sequence {name_stream}\n")
        strategy.eval(getattr(test_benchmark, name_stream),
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=persistent_workers)