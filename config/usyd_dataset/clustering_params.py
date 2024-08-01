def get_clustering_params():
    params = {
        "type": 0,
        "restart_after_each_sequence": False,
        "dataset_type": "kitti",
        "desired_average_cardinality": 1000,
        "mesh_shape_factor": 1,
        "radius_upper_bound": 100,
        "dynamic": True,  #the clustering is dynamic or not
        "max_distance_for_link": 10, #m, negative to set infinite distance
        "max_links": 1, #negative to set no max
        "link_each_sequence": True #if true, links are added to all the #max_links nns in each previous sequences, otherwise only to the #max_links nns in the cumulative sequence
    }
    
    return params

__all__ = ["get_clustering_params"]