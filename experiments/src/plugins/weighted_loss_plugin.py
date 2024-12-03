from typing import Callable, TYPE_CHECKING

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
    
from .replay_plugin import ReplayPlugin 
from memory_profiler import profile
import globals

class WeightedLossPlugin(SupervisedPlugin):
    """
    Plugin to weigh the loss function depending on the classes distribution in the dataset.
    """
    def __init__(
        self,
        compute_train_weights_fn: Callable = None, #function to compute the weights
        compute_valid_weights_fn: Callable = None, #function to compute the weights
    ):
        super().__init__()
        self.compute_train_weights_fn = compute_train_weights_fn
        self.compute_valid_weights_fn = compute_valid_weights_fn

    @profile(stream=globals.memory_profiler_file)
    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if self.compute_train_weights_fn is not None:
            if isinstance(strategy.plugins[0], ReplayPlugin):
                total_dataset = strategy.adapted_dataset + strategy.plugins[0].storage_policy.buffer
            else:
                total_dataset = strategy.experience.dataset
            
            targets = torch.tensor([e[2] for e in total_dataset], device = strategy.device)    
             
            classes, counts = torch.unique(targets, return_counts=True)
            print(counts)
            print(classes)
            samples_per_class = torch.zeros(torch.count_nonzero(strategy.model.classifier.last_layer.active_units), device=strategy.device)
            samples_per_class[classes.long().to(strategy.device, non_blocking=True)] = counts.float().to(strategy.device, non_blocking=True)
            
            print(samples_per_class)
            
            weights = self.compute_train_weights_fn(samples_per_class)
            if globals.debug:
                with open(globals.debug_file, "a") as file:
                    file.write(f"Samples per class: {samples_per_class.tolist()}\n")
                    file.write(f"Weights: {weights.tolist()}\n")
            strategy.train_classifier_criterion.weight = weights
        
    def before_eval_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if self.compute_valid_weights_fn is not None:
            targets = torch.tensor([e[2] for e in strategy.experience.dataset], device = strategy.device)
            classes, counts = torch.unique(targets, return_counts=True)
            samples_per_class = torch.zeros(strategy.model.classifier.last_layer.classifier.out_features, device=strategy.device)
            samples_per_class[classes.long().to(strategy.device, non_blocking=True)] = counts.float().to(strategy.device, non_blocking=True)
            weights = self.compute_valid_weights_fn(samples_per_class)
            if globals.debug:
                with open(globals.debug_file, "a") as file:
                    file.write(f"Samples per class: {samples_per_class.tolist()}\n")
                    file.write(f"Weights: {weights.tolist()}\n")
            strategy.eval_classifier_criterion.weight = weights
            
    