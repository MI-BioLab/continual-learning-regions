def get_clustering_params():
    params = {
        "type": 1,
        "restart_after_each_sequence": False,
        "dataset_type": "openloris",
        "desired_average_cardinality": 100,
        "mesh_shape_factor": 1,
        "radius_upper_bound": 15,
        "dynamic": True, #the clustering is dynamic or not
        "max_distance_for_link": 2, #m, negative to set infinite distance
        "max_links": 1, #negative to set no max
        "link_each_sequence": True, #if true, links are added to all previous sequences
        "max_links_intra_sequence": 1, #negative to set no max
        "valid_node_every": 1, #min timestamp difference in seconds between valid nodes
        "min_distance_for_valid_node": 0.1, #min distance in meters between valid nodes
        "min_time_for_link": 60, #min time in seconds between not adjacent nodes to add a link in current sequence (for loop closure detection)
        "fps": 30,
        "time_multiplier": 1 #1.e+9    
    }
    
    return params

__all__ = ["get_clustering_params"]