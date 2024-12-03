import os

def get_dataset_params():
    root = "/mnt/e/StLucia" #TODO CHANGE
    train_sequences = ["100909_0845",
                       "100909_1000",
                       "100909_1210",
                       "100909_1410",
                       "110909_1545",
                       "180809_1545"]
    
    test_sequences = ["190809_0845",
                      "190809_1410",
                      "210809_1000",
                      "210809_1210"]
    
    streams = ["front"]
    images_folders = ["frames"]

    
    params = {
        "load_dataset": False,
        "dataset_folder": "./res/datasets/stlucia",
        "image_width": 224,
        "image_height": 224,
        "use_validation": True, #for offline experiments
        # "save_tensor_dataset": True,
        # "tensor_dataset_folders": [os.path.join(root, "tensors", s) for s in streams],
        # "load_tensor_dataset": False,
        "image_streams": streams,
        "images_folders": images_folders,
        "train": {
            "sequences": [os.path.join(root, s) for s in train_sequences],
            "poses_files": ["positions1.txt"] * len(train_sequences),
            "experience_size": 600
        },
        "validation": { #for offline experiments
            "balanced_validation": True,
            "validation_size": 0.1
        },
        "test": {
            "sequences": [os.path.join(root, s) for s in test_sequences],
            "poses_files": ["positions1.txt"] * len(test_sequences),
        }
    }
    
    return params
    
__all__ = ["get_dataset_params"]