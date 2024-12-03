import os

def get_dataset_params():
    root = "/mnt/e/OpenLoris/market"
    train_sequences = ["market2",
                       "market3"]
    
    test_sequences = ["market1"]
    
    streams = ["front"]
    images_folders = ["color_224"]

    
    params = {
        "load_dataset": True,
        "dataset_folder": "./res/datasets/openloris/market",
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
            "poses_files": ["aligned_groundtruth.txt"] * len(train_sequences),
            "experience_size": 600
        },
        "validation": { #for offline experiments
            "balanced_validation": True,
            "validation_size": 0.1
        },
        "test": {
            "sequences": [os.path.join(root, s) for s in test_sequences],
            "poses_files": ["aligned_groundtruth.txt"] * len(test_sequences),
        }
    }
    
    return params
    
__all__ = ["get_dataset_params"]