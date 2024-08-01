import os


def get_continual_params(**kwargs):
    save_folder = "./res/continual/usyd"
    params = {
        "seed": 42,
        "save_folder": save_folder,
        "restore_from_checkpoint": False,  #TODO CHANGE
        "checkpoint_folder": os.path.join(save_folder,
                                          "experiment_0"),  #TODO CHANGE
        "strategy": {
            "type": 0,
            "streams": [1],
            "sum_predictions": True,
            "stream_weights": [1.] * 3
        },
        "model": {
            "name": "resnet18",
            "state_dict_file":
            "../../places365/resnet18_places365/resnet18_places365.pth",
            "trainable_from_layer": "layer4",
            "incremental": True
        },
        "plugins": {
            "one_hot_encoder_plugin": None,
            "replay_plugin": {
                "replay_memory_size": 500,
                "batch_size_mem": 64,
                "storage_policy": 1
            },
            "update_replay_memory_plugin": {
                "remove_unused": True,
                "update": True
            },
            "feature_extractor_plugin": {
                "batch_size": 256
            }
        },
        "train": {
            "batch_size": 64,
            "epochs": 4,
            "initial_learning_rate": 1.e-4,
            "optimizer": {
                "type": 1
                # momentum: 0.9
            },
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
        "test": {
            "batch_size": 256,
            "test_every": 1,
            "loss": {
                "loss_fn": 0
            }
        }
    }
    
    return params
