from typing import Sequence, Optional, TypeVar

from avalanche.training.templates.common_templates import SupervisedTemplate
from torch.nn import Module
from torch.optim import Optimizer
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.benchmarks.scenarios.generic_scenario import DatasetExperience
from avalanche.core import BaseSGDPlugin

import sys
sys.path.insert(0, "src/")
from setting_strategies import BaseStrategy
import globals

TDatasetExperience = TypeVar('TDatasetExperience', bound=DatasetExperience)
class ContinualStrategy(SupervisedTemplate):
    def __init__(
            self,
            setting_strategy: BaseStrategy,
            nodes_moved_in_each_experience: Sequence,
            optimizers: Sequence[Optimizer],
            criterions: Sequence[Module],
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device="cuda",
            plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
            evaluator=default_evaluator(),
            eval_every=-1,
            peval_mode="experience",
            **kwargs
    ):

        self.setting_strategy = setting_strategy
        self.optimizer = optimizers[0]
        self.train_classifier_criterion, self.eval_classifier_criterion = criterions
        self.num_classes = self.setting_strategy.model.classifier.last_layer.classifier.out_features
        self.nodes_moved_in_each_experience = nodes_moved_in_each_experience
        
        super().__init__(
            model=self.setting_strategy.model,
            optimizer=self.optimizer,
            criterion=self.train_classifier_criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )        
        
    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        self.setting_strategy.model.classifier.last_layer = super().model_adaptation(model=self.setting_strategy.model.classifier.last_layer)
        self.num_classes = self.setting_strategy.model.classifier.last_layer.classifier.out_features
        if globals.debug:
            with open(globals.debug_file, "a") as file:
                file.write(f"Model adaptation\n")
        return self.setting_strategy.model            
    
    @property
    def mb_x(self):
        """Current mini-batch input."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[1]
    
    @property
    def mb_y(self):
        """Current mini-batch target."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[2]
    
    @property
    def mb_id(self):
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[0]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3
        return mbatch[-1]
    
    def forward(self):
        features = self.setting_strategy.extract_trainable_features(self.mb_x)
        input = self.setting_strategy.after_trainable_features_extraction(features)
        predictions = self.setting_strategy.classify(input)
        output = self.setting_strategy.after_classification(predictions)
        if globals.debug:
            with open(globals.debug_file, "a") as file:
                file.write(f"Forward\n")
        return output
    
    def _before_training_exp(self, **kwargs):
        self._criterion = self.train_classifier_criterion
        return super()._before_training_exp(**kwargs)
        
    def _before_eval_exp(self, **kwargs):
        self._criterion = self.eval_classifier_criterion
        return super()._before_eval_exp(**kwargs)
        
    def _after_eval(self, **kwargs):
        self._criterion = self.train_classifier_criterion
        return super()._after_eval(**kwargs)
    
    def _unpack_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3

        if isinstance(mbatch, tuple):
            mbatch = list(mbatch)
        self.mbatch[1] = mbatch[1].to(self.device, non_blocking=True) #images
        self.mbatch[2] = mbatch[2].to(self.device, non_blocking=True) #targets
        
        