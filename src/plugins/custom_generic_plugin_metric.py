"""
This is a customized version of Avalanche GenericPluginMetric to enable logging
of metrics using the experience ID as the X axis instead of the train iteration.
"""

from typing import TypeVar, TYPE_CHECKING
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricResult
    from avalanche.training.templates import SupervisedTemplate

from avalanche.evaluation.metric_definitions import PluginMetric

TResult = TypeVar("TResult")
TAggregated = TypeVar("TAggregated", bound="PluginMetric")

"""
Modified version of the GenericPluginMetric of Avalanche to compute the metrics at experience level instead of iteration level.
"""
class GenericPluginMetric(PluginMetric[TResult]):
    """
    This class provides a generic implementation of a Plugin Metric.
    The user can subclass this class to easily implement custom plugin
    metrics.
    """

    def __init__(
        self, metric, reset_at="experience", emit_at="experience", mode="eval", experience_granularity=False
    ):
        super(GenericPluginMetric, self).__init__()
        assert mode in {"train", "eval"}
        if mode == "train":
            assert reset_at in {"iteration", "epoch", "experience", "stream",
                                "never"}
            assert emit_at in {"iteration", "epoch", "experience", "stream"}
        else:
            assert reset_at in {"iteration", "experience", "stream", "never"}
            assert emit_at in {"iteration", "experience", "stream"}
        self._metric = metric
        self._reset_at = reset_at
        self._emit_at = emit_at
        self._mode = mode
        self._experience_granularity = experience_granularity

    def reset(self, strategy) -> None:
        self._metric.reset()

    def result(self, strategy):
        return self._metric.result()

    def update(self, strategy):
        pass

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result(strategy)
        add_exp = self._emit_at == "experience"
        if self._experience_granularity:
            plot_x_position = strategy.clock.train_exp_counter
        else:
            plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k
                )
                metrics.append(
                    MetricValue(self, metric_name, v, plot_x_position)
                )
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            return [
                MetricValue(self, metric_name, metric_value, plot_x_position)
            ]

    def before_training(self, strategy: "SupervisedTemplate"):
        super().before_training(strategy)
        if self._reset_at == "stream" and self._mode == "train":
            self.reset()

    def before_training_exp(self, strategy: "SupervisedTemplate"):
        super().before_training_exp(strategy)
        if self._reset_at == "experience" and self._mode == "train":
            self.reset(strategy)

    def before_training_epoch(self, strategy: "SupervisedTemplate"):
        super().before_training_epoch(strategy)
        if self._reset_at == "epoch" and self._mode == "train":
            self.reset(strategy)

    def before_training_iteration(self, strategy: "SupervisedTemplate"):
        super().before_training_iteration(strategy)
        if self._reset_at == "iteration" and self._mode == "train":
            self.reset(strategy)

    def after_training_iteration(self, strategy: "SupervisedTemplate") -> None:
        super().after_training_iteration(strategy)
        if self._mode == "train":
            self.update(strategy)
        if self._emit_at == "iteration" and self._mode == "train":
            return self._package_result(strategy)

    def after_training_epoch(self, strategy: "SupervisedTemplate"):
        super().after_training_epoch(strategy)
        if self._emit_at == "epoch" and self._mode == "train":
            return self._package_result(strategy)

    def after_training_exp(self, strategy: "SupervisedTemplate"):
        super().after_training_exp(strategy)
        if self._emit_at == "experience" and self._mode == "train":
            return self._package_result(strategy)

    def after_training(self, strategy: "SupervisedTemplate"):
        super().after_training(strategy)
        if self._emit_at == "stream" and self._mode == "train":
            return self._package_result(strategy)

    def before_eval(self, strategy: "SupervisedTemplate"):
        super().before_eval(strategy)
        if self._reset_at == "stream" and self._mode == "eval":
            self.reset(strategy)

    def before_eval_exp(self, strategy: "SupervisedTemplate"):
        super().before_eval_exp(strategy)
        if self._reset_at == "experience" and self._mode == "eval":
            self.reset(strategy)

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        super().after_eval_exp(strategy)
        if self._emit_at == "experience" and self._mode == "eval":
            return self._package_result(strategy)

    def after_eval(self, strategy: "SupervisedTemplate"):
        super().after_eval(strategy)
        if self._emit_at == "stream" and self._mode == "eval":
            return self._package_result(strategy)

    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        super().after_eval_iteration(strategy)
        if self._mode == "eval":
            self.update(strategy)
        if self._emit_at == "iteration" and self._mode == "eval":
            return self._package_result(strategy)

    def before_eval_iteration(self, strategy: "SupervisedTemplate"):
        super().before_eval_iteration(strategy)
        if self._reset_at == "iteration" and self._mode == "eval":
            self.reset(strategy)