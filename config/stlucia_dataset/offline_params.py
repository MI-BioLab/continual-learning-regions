import os


def get_offline_params(**kwargs):
    save_folder = "./res/offline/stlucia"
    params = {
        "seed": 42,
        "save_folder": save_folder,
        "restore_from_checkpoint": False,  #TODO CHANGE
        "checkpoint_folder": os.path.join(save_folder,
                                          "experiment_0"),  #TODO CHANGE
        "strategy": {
            "type": 0,
            "streams": [0],
            "sum_predictions": True,
            "stream_weights": [1.] * 3
        },
        "model": {
            "model_name": "resnet18",
            "state_dict_file":
            "../../places365/resnet18_places365/resnet18_places365.pth",
            "trainable_from_layer": "layer4",
            "incremental": False
        },
        "train": {
            "batch_size": 256,
            "epochs": 100,
            "initial_learning_rate": 1.e-4,
            "use_learning_rate_decay": False,
            "patience_for_learning_rate_decay": 5,  # learning rate decay policy
            "learning_rate_multiplier": 0.3333,
            "use_early_stopping": False,
            "min_learning_rate": 1.e-6,
            "balance_minibatches": False,
            "replacement": True,
            "minibatches_weighting": {
                "weighting_method": 0
            },
            "optimizer": {
                "type": 1,
                # momentum: 0.9
            },
            "save_best_model": True,
            "save_end_model": True,
            "loss": {
                "loss_fn": 1,
                "gamma": 2,
            },
            "loss_weighting": {
                "weighting_method": 1,
                "beta": 0.999
            }
        },
        "validation": {
            "balanced_validation": True,
            "validation_size": 0.1,
            "batch_size": 256,
            "loss": {
                "loss_fn": 0
            },  
        },
        "test": {
            "batch_size": 256,
            "loss": {
                "loss_fn": 0
            },
        }
    }

    return params


__all__ = ["get_offline_params"]
