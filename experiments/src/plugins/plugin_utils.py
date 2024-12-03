import os
import torch 

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (forgetting_metrics, 
                                          confusion_matrix_metrics, 
                                          loss_metrics, 
                                          timing_metrics, 
                                          cpu_usage_metrics, 
                                          gpu_usage_metrics, 
                                          ram_usage_metrics)
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from plugins import FeatureExtractionPlugin, OneHotEncoderPlugin, WeightedLossPlugin, ReplayPlugin, UpdateReplayMemoryPlugin

from avalanche.evaluation.metrics.topk_acc import topk_acc_metrics
from metrics import custom_accuracy_metrics
from utils import get_storage_policy

def get_plugins(params, return_eval=True, **kwargs):
    # plugins
    train_plugins = []
    if "plugins" in params:
        plugin_params = params["plugins"]
    else:
        plugin_params = dict()
        
    if "replay_plugin" in plugin_params:
        replay_plugin = ReplayPlugin(
            mem_size=plugin_params["replay_plugin"]["replay_memory_size"],
            batch_size=params["train"]["batch_size"],
            batch_size_mem=plugin_params["replay_plugin"]
            ["batch_size_mem"],
            storage_policy=get_storage_policy(
                plugin_params["replay_plugin"]["storage_policy"],
                **plugin_params["replay_plugin"]))
        train_plugins.append(replay_plugin)

        
        if "update_replay_memory_plugin" in plugin_params:
            rm_params = plugin_params["update_replay_memory_plugin"]
            remove_unused = rm_params["remove_unused"] \
                if "remove_unused" in rm_params and rm_params["remove_unused"] is not None \
                    else False
            update = rm_params["update"] \
                if "update" in rm_params and rm_params["update"] is not None \
                    else False
            update_replay_memory_plugin = UpdateReplayMemoryPlugin(
                plugin_params["replay_plugin"]["storage_policy"],
                remove_unused,
                update
                )
            train_plugins.append(update_replay_memory_plugin)

    #mandatory
    feature_extraction_plugin = FeatureExtractionPlugin(
        plugin_params["feature_extractor_plugin"]["batch_size"])
    train_plugins.append(feature_extraction_plugin)

    #mandatory
    one_hot_encoder_plugin = OneHotEncoderPlugin()
    train_plugins.append(one_hot_encoder_plugin)

    if "loss_weighting" in params["train"] or "loss_weighting" in params[
            "validation"] or "loss_weighting" in params["test"]:
        weighted_loss_plugin = WeightedLossPlugin(kwargs["compute_train_weights_fn"],
                                                  kwargs["compute_eval_weights_fn"])
        train_plugins.append(weighted_loss_plugin)

    eval_plugin = None
    if return_eval:
        eval_plugin = [
            topk_acc_metrics(top_k=1,
                                minibatch=True,
                                epoch=True,
                                experience=True,
                                stream=True),
            topk_acc_metrics(top_k=3,
                                minibatch=True,
                                epoch=True,
                                experience=True,
                                stream=True),
            topk_acc_metrics(top_k=5,
                                minibatch=True,
                                epoch=True,
                                experience=True,
                                stream=True),
            custom_accuracy_metrics(minibatch=True,
                                    epoch=True,
                                    experience=True,
                                    stream=True,
                                    experience_granularity=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True),
            forgetting_metrics(experience=True, stream=True),
            # cpu_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            # gpu_usage_metrics(torch.cuda.current_device(), minibatch=True, epoch=True, experience=True, stream=True),
            # ram_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            
        ]
        if params["train"]["full_test_set"]:
            eval_plugin.append(
                confusion_matrix_metrics(num_classes=kwargs["num_classes"],
                                        save_image=True,
                                        stream=True,
                                        normalize="true"),
            )
        eval_plugin = EvaluationPlugin(
            *eval_plugin,
            loggers=[
                InteractiveLogger(),
                TensorboardLogger(os.path.join(kwargs["new_folder"], "tb_data_" + "full_test_set" if params["train"]["full_test_set"] else "seen_so_far")),
                TextLogger(open(os.path.join(kwargs["new_folder"], "logs.txt"), "a"))
            ],
            collect_all=False,
            strict_checks=False
            )

    return train_plugins, eval_plugin 

__all__ = ["get_plugins"]