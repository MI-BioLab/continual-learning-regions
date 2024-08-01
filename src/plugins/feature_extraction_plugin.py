from typing import TYPE_CHECKING

from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.benchmarks.utils import make_classification_dataset

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from setting_strategies import BaseStrategy
import globals
from memory_profiler import profile

"""
Plugin to extract and concatenate features from images before training.
"""
class FeatureExtractionPlugin(SupervisedPlugin):
    def __init__(
        self,
        feature_extraction_batch_size=256,
    ):
        super().__init__()
        self.feature_extraction_batch_size = feature_extraction_batch_size

    @profile(stream=globals.memory_profiler_file)
    def _extract_features(self, dataset, setting_strategy: BaseStrategy):
        if globals.debug:
            with open(globals.debug_file, "a") as file:
                file.write(f"Feature extraction\n")
        dataloader = DataLoader(dataset,
                                self.feature_extraction_batch_size,
                                shuffle=False,
                                num_workers=8,
                                persistent_workers=False,
                                pin_memory=True)

        freezed_features = []
        ids = []
        targets = []

        for mini_batch in tqdm(dataloader):  #(id, [images], targets, task)
            input = setting_strategy.before_feature_extraction(mini_batch).to(
                setting_strategy.model.device, non_blocking=True)
                                    
            freezed_f = setting_strategy.extract_freezed_features(input)
            freezed_f = setting_strategy.after_freezed_features_extraction(freezed_f)
            freezed_features.append(freezed_f.detach().cpu())
            ids.extend(mini_batch[0])
            targets.extend(mini_batch[2])
       
        freezed_features = torch.cat(freezed_features, dim=0)            
        ids = torch.tensor(ids)
        targets = torch.tensor(targets)
 
        experience_dataset = TensorDataset(ids, freezed_features, targets)  
        return make_classification_dataset(experience_dataset, task_labels=[0] * len(experience_dataset), targets=targets) #targets)

    def after_train_dataset_adaptation(self, 
                                       strategy: "SupervisedTemplate",
                                       **kwargs):
        strategy.adapted_dataset = self._extract_features(
            strategy.adapted_dataset, strategy.setting_strategy)

    def after_eval_dataset_adaptation(self, 
                                      strategy: "SupervisedTemplate",
                                      **kwargs):
        strategy.adapted_dataset = self._extract_features(
            strategy.adapted_dataset, strategy.setting_strategy)
