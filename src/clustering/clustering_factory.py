from .clustering_from_graph import ClusteringFromGraph
from .clustering_from_groundtruth import ClusteringFromGroundtruth

def get_clustering(value, **kwargs):
    match value:
        case 0:
            return ClusteringFromGraph(kwargs["desired_average_cardinality"],
                                       kwargs["mesh_shape_factor"],
                                       kwargs["radius_upper_bound"],
                                       kwargs["max_distance_for_link"],
                                       kwargs["max_links"],
                                       kwargs["link_each_sequence"],
                                       kwargs["restart_after_each_sequence"])
        case 1:
            #TODO restart_after_each_sequence
            return ClusteringFromGroundtruth(kwargs["desired_average_cardinality"],
                                            kwargs["mesh_shape_factor"],
                                            kwargs["radius_upper_bound"],
                                            kwargs["max_distance_for_link"],
                                            kwargs["max_links"],
                                            kwargs["link_each_sequence"],
                                            kwargs["restart_after_each_sequence"],
                                            kwargs["valid_node_every"],
                                            kwargs["min_distance_for_valid_node"],
                                            kwargs["fps"],
                                            kwargs["time_multiplier"],
                                            kwargs["min_time_for_link"],
                                            kwargs["max_links_intra_sequence"])
    raise ValueError("get_clustering: invalid clustering")

__all__ = ["get_clustering"]