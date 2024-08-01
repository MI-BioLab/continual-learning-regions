from typing import TYPE_CHECKING

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
from memory_profiler import profile
import globals

"""
Plugin to encode in/decode from one-hot format the labels.
"""
class OneHotEncoderPlugin(SupervisedPlugin):
    def __init__(
        self,
    ):
        super().__init__()
    
    @profile(stream=globals.memory_profiler_file)
    def _encode_one_hot(self, num_classes, y):
        return torch.nn.functional.one_hot(y, num_classes).float() #one-hot encoding
    
    @profile(stream=globals.memory_profiler_file)
    def _decode_one_hot(self, y):
        return torch.argmax(y, dim=-1) #one-hot decoding

    @profile(stream=globals.memory_profiler_file)
    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.mbatch[2] = self._encode_one_hot(strategy.num_classes, strategy.mb_y.long())

    @profile(stream=globals.memory_profiler_file)
    def after_forward(self, strategy: "SupervisedTemplate", **kwargs):
        idx = torch.argwhere(strategy.model.classifier.last_layer.active_units).squeeze(-1)
        strategy.mb_output = strategy.mb_output[..., idx]
        strategy.mbatch[2] = self._decode_one_hot(strategy.mb_y)
    
    @profile(stream=globals.memory_profiler_file)    
    def before_eval_forward(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.mbatch[2] = self._encode_one_hot(strategy.num_classes, strategy.mb_y.long()) 
    
    @profile(stream=globals.memory_profiler_file)    
    def after_eval_forward(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.mbatch[2] = self._decode_one_hot(strategy.mb_y)
        
        
    