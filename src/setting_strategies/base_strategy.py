from typing import Sequence

from abc import abstractmethod

import torch

"""
Classes to specify the strategy in case of more streams of images.
They must be considered as a TODO, so always a SingleStreamStrategy is used for now.
"""
class BaseStrategy:
    def __init__(self, 
                 model, 
                 **kwargs):
        self.model = model
        
    @abstractmethod 
    def prepare_data(self, ids: Sequence, images: Sequence, targets: Sequence = None):
        raise NotImplementedError

    @abstractmethod
    def before_feature_extraction(self, batch: Sequence) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def extract_freezed_features(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def after_freezed_features_extraction(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def extract_trainable_features(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def after_trainable_features_extraction(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def after_classification(self, predictions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError