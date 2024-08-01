from typing import Sequence, Tuple
import torch

from setting_strategies import BaseStrategy
from models import Model

"""
Classes to specify the strategy in case of more streams of images.
They must be considered as a TODO, so always a SingleStreamStrategy is used for now.
"""
class MultiStreamStrategy(BaseStrategy):
    def __init__(self, 
                 model: Model,
                 sum_predictions: bool = False,
                 stream_weights: torch.Tensor = None,
                 **kwargs):
        super(MultiStreamStrategy, self).__init__(model)
        self.sum_predictions = sum_predictions
        self.stream_weights = stream_weights
        self.streams = kwargs["streams"]
        
    def prepare_data(self, ids: Sequence, paths: Sequence, targets: Sequence = None):
        assert len(paths[0]) > 1 and len(self.streams) > 1, "paths and streams should be more than one"
        assert len(self.streams) <= len(paths[0]), "streams cannot be more then images"
        return ids, [[path[i] for i in self.streams] for path in paths], targets

    def before_feature_extraction(self, batch: Sequence) -> torch.Tensor:
        return torch.stack(batch[1]) #(id, images, target, task_labels), stack all the images
    
    def extract_freezed_features(self, input: torch.Tensor) -> torch.Tensor:
        if self.model.feature_extractor.freezed_part is None:
            return input
        freezed_features = []
        for n in range(input.shape[1]):
            freezed_f = self.model.feature_extractor.extract_freezed_features(input[:, n, ...])
            freezed_features.append(freezed_f)
        return torch.stack(freezed_features, dim=1)
    
    def after_freezed_features_extraction(self, features: torch.Tensor) -> torch.Tensor:
        return features
    
    def extract_trainable_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.model.feature_extractor.trainable_part is None:
            return features
        trainable_features = []
        for n in range(features.shape[1]):
            trainable_f = self.model.feature_extractor.extract_trainable_features(features[:, n, ...])
            trainable_features.append(trainable_f)
        return torch.stack(trainable_features, dim=1)
    
    def after_trainable_features_extraction(self, features: torch.Tensor) -> torch.Tensor:
        if self.stream_weights is None:
            self.stream_weights = torch.tensor(1/features.shape[1]).repeat(features.shape[1]).to(self.model.device, non_blocking=True)
        else:
            self.stream_weights = torch.tensor(self.stream_weights).to(self.model.device, non_blocking=True)
        return features

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        predictions = []
        for feature in features:
        # for n in range(features.shape[1]):
            predictions.append(self.model.classifier(feature))
        return torch.stack(predictions, dim=1)
    
    def after_classification(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.sum_predictions:
            weighted_predictions = torch.einsum("k,ikj->ikj", self.stream_weights, predictions)
            return torch.sum(weighted_predictions, dim=1)
        return predictions