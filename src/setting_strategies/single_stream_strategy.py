from typing import Sequence
import torch

from setting_strategies import BaseStrategy
from models import Model

"""
Classes to specify the strategy in case of more streams of images.
They must be considered as a TODO, so always a SingleStreamStrategy is used for now.
"""
class SingleStreamStrategy(BaseStrategy):
    def __init__(self, 
                 model: Model,
                 **kwargs):
        super(SingleStreamStrategy, self).__init__(model)
        self.streams = kwargs["streams"]
        
    def prepare_data(self, ids: Sequence, paths: Sequence, targets: Sequence = None):
        assert len(self.streams) >= 1 and len(paths[0]) >= 1, "images and streams should be one"
        assert len(self.streams) <= len(paths[0]), "streams cannot be more then images"
        if len(paths[0]) == 1:
            assert self.streams[0] == 0, "cannot specify streams > 0 if there is only one image"
            return ids, [path for path in paths], targets
        else:
            n = len(self.streams)
            if n <= 1:
                return ids, [[path[self.streams[0]]] for path in paths], targets
            
            new_ids = []
            new_paths = []
            new_targets = []
            
            for j in range(len(ids)):
                for i in self.streams:
                    new_ids.append(float(f"{ids[j]}.{i+1}"))
                    new_paths.append(paths[j][i])
                    if targets is not None:
                        new_targets.append(targets[j])
            return new_ids, new_paths, new_targets if len(new_targets) > 0 else targets    
    
    def before_feature_extraction(self, batch: Sequence) -> torch.Tensor:
        return batch[1][0] #single stream, (id, [image], target, task_labels)
    
    def extract_freezed_features(self, input: torch.Tensor) -> torch.Tensor:
        return self.model.feature_extractor.extract_freezed_features(input)
    
    def after_freezed_features_extraction(self, features: torch.Tensor) -> torch.Tensor:
        return features
    
    def extract_trainable_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.feature_extractor.extract_trainable_features(features)
    
    def after_trainable_features_extraction(self, features: torch.Tensor) -> torch.Tensor:
        return features
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.classifier(features)
    
    def after_classification(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions