import os

from . import get_parameters_from_yaml
from . import merge_paths

#TODO add offline params

default_seed = 42
default_restore_from_checkpoint = False
default_image_width = 224
default_image_height = 224
default_save_folder = "./res/continual/"
default_use_validation = True

default_dataset_config_file = "dataset.yaml"
default_clustering_type = 0 #from_graph
default_clustering_config_file = "clustering.yaml"
default_load_dataset = False

default_restart_after_each_sequence = False
default_dataset_type = "kitti"
default_desired_average_cardinality = 1000
default_mesh_shape_factor = 1
default_radius_upper_bound = 100 
default_dynamic = True #the clustering is dynamic or not
default_max_distance_for_link = 10 #m, 0 or negative to set infinite distance
default_max_links = 1 #0 or negative to set no max
default_link_each_sequence = True #if true, links are added to all the #max_links nns in each previous sequences, otherwise only to the #max_links nns in the cumulative sequence
default_max_links_intra_sequence = 1 #0 or negative to set no max
default_valid_node_every = 1 #min timestamp difference in seconds between valid nodes
default_min_distance_for_valid_node = 0.1 #min distance in meters between valid nodes
default_min_time_for_link = 10 #min time in seconds between not adjacent nodes to add a link in current sequence (for loop closure detection)
default_fps = 30
default_time_multiplier = 1 #1.e+9


default_setting_strategy = 0 #single stream
default_streams = [0] #first
default_sum_predictions = True

default_model_name = "resnet18"
default_trainable_from_layer = None
default_incremental = False

default_replay_memory_size = 500 
default_batch_size_mem = 64
default_storage_policy = 1
default_feature_extractor_batch_size = 256

default_experience_size = 600
default_batch_size = 256
default_epochs = 100
default_initial_learning_rate = 1e-4
default_use_learning_rate_decay = False
default_patience_for_learning_rate_decay =  3
default_learning_rate_multiplier = 0.3333
default_use_early_stopping = False
default_min_learning_rate = 1e-5
default_balance_minibatches = False
default_minibatches_weighting_method = 0
default_replacement = True
default_optimizer_type = 0 #SGD
default_save_best_model = True
default_save_end_model = True


default_save_end_model = True

#loss parameters (also in test)
default_loss_fn = 0 #cross entropy
default_gamma = 2

default_weighting_method = 0 #simple weights
default_beta = 0.999

default_balanced_validation = True
default_validation_size = 0.1

default_test_every = -1

default_current_data_filename = "example.json"
default_predictions_filename = "predictions.json"
default_experience_filename = "experience.json"
default_db_name = "rtabmap.db"
default_splitting = 0
default_alpha = 0.5

def assign_param(params,
                 name,
                 default_value = None, 
                 check_none = True):
    params[name] = params.get(name, default_value) if check_none and params.get(name) is None else params.get(name, default_value)
    
def assert_param(params,
                 name, 
                 message,
                 check_none = True,
                 check_len = True):
    assert name in params and (params[name] is not None if check_none else True) and (len(params[name]) > 0 if check_len else True), message
    

def parse_clustering_params(config_dir, parameters):
    assert_param(parameters, "clustering", "clustering params must be set", check_none=True, check_len=False)
    clustering_params = parameters["clustering"]
    assign_param(clustering_params, "type", default_clustering_type, check_none=True)
    assign_param(clustering_params, "clustering_config_file", default_clustering_config_file, check_none=True)
                    
    clustering_params = {**clustering_params, **get_parameters_from_yaml(os.path.join(config_dir, clustering_params["clustering_config_file"]))}
    
    assign_param(clustering_params, "restart_after_each_sequence", default_restart_after_each_sequence, check_none=True)
    assign_param(clustering_params, "dataset_type", default_dataset_type, check_none=True)
    assign_param(clustering_params, "desired_average_cardinality", default_desired_average_cardinality, check_none=True)
    assign_param(clustering_params, "mesh_shape_factor", default_mesh_shape_factor, check_none=True)
    assign_param(clustering_params, "radius_upper_bound", default_radius_upper_bound, check_none=True)
    assign_param(clustering_params, "dynamic", default_dynamic, check_none=True)
    assign_param(clustering_params, "max_distance_for_link", default_max_distance_for_link, check_none=True)
    assign_param(clustering_params, "max_links", default_max_links, check_none=True)
    assign_param(clustering_params, "link_each_sequence", default_link_each_sequence, check_none=True)
    assign_param(clustering_params, "max_links_intra_sequence", default_max_links_intra_sequence, check_none=True)
    assign_param(clustering_params, "valid_node_every", default_valid_node_every, check_none=True)
    assign_param(clustering_params, "min_distance_for_valid_node", default_min_distance_for_valid_node, check_none=True)
    assign_param(clustering_params, "min_time_for_link", default_min_time_for_link, check_none=True)
    assign_param(clustering_params, "fps", default_fps, check_none=True)
    assign_param(clustering_params, "time_multiplier", default_time_multiplier, check_none=True)
    
    #TODO clustering from groundtruth params
    
    parameters["clustering"] = clustering_params

def parse_dataset_train_validation_test(parameters, name):
    assert name == "train" or name == "validation" or name == "test", "name must be train, validation or test"
    params = parameters[name]
    if name == "train" or name == "test":
        assert_param(params, "sequences", f"dataset {name} sequences must be specified", check_none=True, check_len=True)
        assert_param(params, "image_streams", f"dataset {name} image_streams must be specified", check_none=True, check_len=True)    

        assert "poses_files" in params and params["poses_files"] is not None and len(params["poses_files"]) > 0, f"{name} poses_files must be specified"
        
        assert len(params["poses_files"]) == len(params["sequences"]), f"dataset {name} poses_files and sequences must have the same len" 
        
        if "ids_files" in params and params["ids_files"] is not None:
            assert len(params["ids_files"]) == len(params["sequences"]), f"dataset {name} ids_files and sequences must have the same len"
        
        if parameters["clustering"]["type"] == 0:
            assert_param(params, "nodes_files", f"dataset {name} nodes_files must be specified", check_none=True, check_len=True)
            assert_param(params, "links_files", f"dataset {name} links_files must be specified", check_none=True, check_len=True)
            assert len(params["nodes_files"]) == len(params["sequences"]), f"dataset {name} nodes_files and sequences must have the same len"
            assert len(params["links_files"]) == len(params["sequences"]), f"dataset {name} links_files and sequences must have the same len"
        
    if name == "train":
        assign_param(params, "experience_size", default_experience_size, check_none=True)
    if name == "validation":
        assign_param(params, "balanced_validation", default_balanced_validation, check_none=True) 
        assign_param(params, "validation_size", default_validation_size, check_none=True) 

def parse_dataset_params(config_dir, parameters):
    assert_param(parameters, "dataset", "dataset params must be set", check_none=True, check_len=False)
    dataset_params = parameters["dataset"]
    assign_param(dataset_params, "load_dataset", default_load_dataset, check_none=True) 
    if parameters["restore_from_checkpoint"]:
        dataset_params["load_dataset"] = True
    if dataset_params["load_dataset"]:
        assert_param(dataset_params, "load_dataset", "dataset load_dataset is true, but no folder is specified", check_none=True, check_len=False)
    assign_param(dataset_params, "dataset_config_file", default_dataset_config_file, check_none=True) 
    
    dataset_config_params = get_parameters_from_yaml(os.path.join(config_dir, dataset_params["dataset_config_file"]))
    parse_clustering_params(config_dir, dataset_config_params)
        
    assert_param(dataset_config_params, "train", "dataset train params must be set", check_none=True, check_len=False)
    assert_param(dataset_config_params, "test", "dataset test params must be set", check_none=True, check_len=False) 
    if "validation" not in dataset_config_params:
        dataset_config_params["validation"] = dict()
    
    parse_dataset_train_validation_test(dataset_config_params, "train")
    parse_dataset_train_validation_test(dataset_config_params, "validation")
    parse_dataset_train_validation_test(dataset_config_params, "test")
    
    assign_param(dataset_params, "image_width", default_image_width, check_none=True)
    assign_param(dataset_params, "image_height", default_image_height, check_none=True)
    assign_param(dataset_params, "use_validation", default_use_validation, check_none=True)
    
    parameters["dataset"] = {**dataset_params, **dataset_config_params}
    
    if "stream_weights" in parameters["strategy"] and parameters["strategy"]["stream_weights"] is not None and parameters["strategy"]["type"] == 1:
        assert len(parameters["strategy"]["stream_weights"]) == len(parameters["dataset"]["train"]["image_streams"]), "stream weights should be the same number as image streams" 
        assert len(parameters["strategy"]["stream_weights"]) == len(parameters["dataset"]["test"]["image_streams"]), "stream weights should be the same number as image streams" 

def parse_strategy_params(parameters):
    if "strategy" not in parameters:
        parameters["strategy"] = dict()
    strategy_params = parameters["strategy"]
    assign_param(strategy_params, "type", default_setting_strategy, check_none=True)
    assign_param(strategy_params, "streams", default_streams, check_none=True)
    if strategy_params["type"] == 1:
        assert len(strategy_params["streams"]) > 1, "streams must be more than 1 if MultiStreamStrategy is used"
        assign_param(strategy_params, "sum_predictions", default_sum_predictions, check_none=True)
        assign_param(strategy_params, "stream_weights", None, check_none=False)
    

def parse_model_params(parameters):
    assert_param(parameters, "model", "model must be specified", check_none=True, check_len=False)
    model_params = parameters["model"]
    assert_param(model_params, "state_dict_file", "model state_dict_file must be specified", check_none=True, check_len=False)
    assign_param(model_params, "model_name", default_model_name, check_none=True)
    assign_param(model_params, "trainable_from_layer", default_trainable_from_layer, check_none=True)
    assign_param(model_params, "incremental", default_incremental, check_none=True)
    
def parse_rtabmap_params(params):
    assign_param(params, "root", None, check_none=False)
    assign_param(params, "current_data_filename", default_current_data_filename, check_none=True)
    assign_param(params, "predictions_filename", default_predictions_filename, check_none=True)
    assign_param(params, "experience_filename", default_experience_filename, check_none=True)
    assign_param(params, "db_name", default_db_name, check_none=True)
    assign_param(params, "streams", None, check_none=True)
    assign_param(params, "splitting", default_splitting, check_none=True)
    assign_param(params, "alpha", default_alpha, check_none=True)
    
    if params["streams"] is not None and len(params["streams"]) > 1:
        if "stream_weights" in params["strategy"]:
            if len(params["strategy"]["stream_weights"]) != len(params["streams"]):
                params["strategy"]["stream_weights"] = None
        else:
            params["strategy"]["stream_weights"] = None
            
def parse_train_validation_test(parameters, 
                                name = "train"):
    assert name == "train" or name == "validation" or name == "test", "name must be train, validation or test"
    params = parameters[name]
    
    assign_param(params, "batch_size", default_batch_size, check_none=True)
    
    if name == "train":
        assign_param(params, "epochs", default_epochs, check_none=True)
        assign_param(params, "initial_learning_rate", default_initial_learning_rate, check_none=True)
        
        assign_param(params, "use_learning_rate_decay", default_use_learning_rate_decay, check_none=True)
        assign_param(params, "patience_for_learning_rate_decay", default_patience_for_learning_rate_decay, check_none=True)
        assign_param(params, "learning_rate_multiplier", default_learning_rate_multiplier, check_none=True)
        assign_param(params, "use_early_stopping", default_use_early_stopping, check_none=True)
        assign_param(params, "min_learning_rate", default_min_learning_rate, check_none=True)
        assign_param(params, "balance_minibatches", default_balance_minibatches, check_none=True)
        assign_param(params, "minibatches_weighting_method", default_minibatches_weighting_method, check_none=True)
        assign_param(params, "replacement", default_replacement, check_none=True)
    
        if "optimizer" not in params:
            params["optimizer"] = dict()
        assign_param(params["optimizer"], "type", default_optimizer_type, check_none=True)
        
        assign_param(params, "save_best_model", default_save_best_model, check_none=True) 
        assign_param(params, "save_end_model", default_save_end_model, check_none=True)    
    
    if "loss" not in params:
        params["loss"] = dict()
    
    assign_param(params["loss"], "loss_fn", default_loss_fn, check_none=True) 
    assign_param(params["loss"], "gamma", default_gamma, check_none=True) 
    if params["loss"]["loss_fn"] == 1 and "loss_weighting" not in params:
        params["loss_weighting"] = dict()
        
    if "loss_weighting" in params:
        assign_param(params["loss_weighting"], "weighting_method", default_weighting_method, check_none=True) 
        assign_param(params["loss_weighting"], "beta", default_beta, check_none=True) 
        
    if name == "validation":
        assign_param(params, "batch_size", default_batch_size, check_none=True) 

    if name == "test":
        assign_param(params, "test_every", default_test_every, check_none=True) 

def parse_common_params(config_dir, parameters):   
    assert_param(parameters, "train", "train params must be set", check_none=True, check_len=False)
    assert_param(parameters, "test", "test params must be set", check_none=True, check_len=False) 
    if "validation" not in parameters:
        parameters["validation"] = dict()   
    
    assign_param(parameters, "seed", default_seed, check_none=True)
    assign_param(parameters, "save_folder", default_save_folder, check_none=True)
    assign_param(parameters, "restore_from_checkpoint", default_restore_from_checkpoint, check_none=True)
    
    if parameters["restore_from_checkpoint"]:
        assert_param(parameters, "restore_from_checkpoint_file", 
                     "restore_from_checkpoint is true but no restore_from_checkpoint_file is specified", 
                     check_none=True, 
                     check_len=False)
        restore_params = get_parameters_from_yaml(os.path.join(config_dir, parameters["restore_from_checkpoint_file"]))
        parameters["save_folder"] = restore_params["save_folder"]
        print(parameters["save_folder"])
    
    #strategy
    parse_strategy_params(parameters)
    
    #model
    parse_model_params(parameters)
    
    parse_train_validation_test(parameters, "train")
    parse_train_validation_test(parameters, "validation")
    parse_train_validation_test(parameters, "test")
    
def parse_params_for_training(config_dir, parameters):
    parse_common_params(config_dir, parameters)
    parse_dataset_params(config_dir, parameters)
                
__all__ = ["merge_paths",
           "assign_param",
           "assert_param",
           "parse_common_params",
           "parse_dataset_params",
           "parse_model_params",
           "parse_rtabmap_params",
           "parse_params_for_training"
           ]    
