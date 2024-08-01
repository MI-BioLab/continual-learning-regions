import os

def get_dataset_params():
    root = "/mnt/e/USyd" #TODO change everytime
    train_sequences = ["Week1_2018-03-08",
                        "Week3_2018-03-22",
                        "Week4_2018-03-27",
                        "Week5_2018-04-05",
                        "Week6_2018-04-10",
                        "Week7_2018-04-19",
                        "Week8_2018-04-27",
                        "Week10_2018-05-09",
                        "Week12_2018-05-28",
                        "Week14_2018-06-13",
                        "Week15_2018-06-18",
                        "Week17_2018-07-05",
                        "Week20_2018-07-24",
                        "Week22_2018-08-10",
                        "Week23_2018-08-15",
                        "Week24_2018-08-24"]
    
    test_sequences = ["Week33_2018-10-22",
                      "Week38_2018-11-27",
                      "Week39_2018-12-17",
                      "Week40_2019-01-08",
                      "Week41_2019-02-01",
                      "Week45_2019-03-04",
                      "Week48_2019-03-27",
                      "Week49_2019-04-01",
                      "Week50_2019-04-11",
                      "Week51_2019-04-16"]
    
    streams = ["left", "front", "right"]    
    images_folders = ["left_224", "front_224", "right_224"]
    
    params = {
        "load_dataset": True,
        "dataset_folder": "./res/datasets/usyd",
        "image_width": 224,
        "image_height": 224,
        "use_validation": True, #for offline experiments
        "image_streams": streams,
        "images_folders": [os.path.join("images", f) for f in images_folders],
        "train": {
            "sequences": [os.path.join(root, s) for s in train_sequences],
            "poses_files": ["poses_aligned_1.txt"] * len(train_sequences),
            "links_files": ["links.txt"] * len(train_sequences),
            "nodes_files": ["nodes.txt"] * len(train_sequences),
            "experience_size": 600
        },
        "validation": { #for offline experiments
            "balanced_validation": True,
            "validation_size": 0.1
        },
        "test": {
            "sequences": [os.path.join(root, s) for s in test_sequences],
            "poses_files": ["poses_aligned_1.txt"] * len(test_sequences),
            "links_files": ["links.txt"] * len(test_sequences),
            "nodes_files": ["nodes.txt"] * len(test_sequences)
        }
    }
    
    return params
    
__all__ = ["get_dataset_params"]