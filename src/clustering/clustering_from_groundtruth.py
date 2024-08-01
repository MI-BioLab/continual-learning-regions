from typing import Sequence
import numpy as np
from scipy.spatial import KDTree

from .base_clustering import BaseClustering
from .link import Link

import globals

"""
Class to perform clustering given a poses ground truth.
Nodes with relative poses must be provided, while links are inserted by the algorithm.
"""
class ClusteringFromGroundtruth(BaseClustering):
    def __init__(self, 
                 desired_average_cardinality, 
                 mesh_shape_factor, 
                 radius_upper_bound,
                 max_distance_for_link,
                 max_links,
                 link_each_sequence,
                 restart_after_each_sequence,
                 valid_node_every,
                 min_distance_for_valid_node,
                 fps,
                 time_multiplier,
                 min_time_for_link,
                 max_links_intra_sequence,
                 current_region_id=0):
        super().__init__(desired_average_cardinality, 
                         mesh_shape_factor, 
                         radius_upper_bound, 
                         max_distance_for_link,
                         max_links,
                         link_each_sequence,
                         restart_after_each_sequence,
                         current_region_id)
        self.valid_node_every = valid_node_every
        self.min_distance_for_valid_node = min_distance_for_valid_node
        self.max_links = max_links
        self.fps = fps
        self.rate = 1. / fps
        self.time_multiplier = time_multiplier
        self.min_time_for_link = min_time_for_link
        self.max_links_intra_sequence = max_links_intra_sequence if max_links_intra_sequence >= 0 else None
        
    """
    Set the current sequence to cluster.
    """
    def set_current_sequence(self, 
                             nodes: Sequence, 
                             links: Sequence = None):     
        super().set_current_sequence(nodes, links)
        self.with_timestamps = self.nodes[0].timestamp is not None
        self.time_since_last_valid_node = 0
    
    """
    Function to clusterize the next node in the sequence.
    """
    def clusterize_next_node(self, dynamic=True):
        
        if self.current_node_idx >= len(self.nodes):
            return False
        
        super().clusterize_next_node(dynamic)
        
        node = self.nodes[self.current_node_idx]
            
        if self.current_node_idx == 0: #new sequence
            if globals.debug:
                with open(globals.clustering_file, "a") as f:
                    f.write(f"New sequence!\n")
            if node.weight != 0: #first node of a sequence always set as valid
                node.weight = 0 
            if len(self.memory.nodes) == 0: #first sequence
                node.region_id = self.current_region_id
                if self.with_timestamps:
                    self.current_timestamp = node.timestamp
                else:
                    self.current_timestamp = 0
                    node.timestamp = self.current_timestamp
                self.memory.nodes[node.id] = node
                self.memory.last_valid_node = node  
                self.current_node_idx += 1  
                return True   
    
        #not first node
        if not self.with_timestamps: #if without timestamps, it is computed with rate (fps)
            self.time_since_last_valid_node += self.rate
            self.current_timestamp += self.rate
            node.timestamp = self.current_timestamp
        else:
            self.time_since_last_valid_node += (abs(node.timestamp - self.nodes[self.current_node_idx - 1].timestamp) * self.time_multiplier)

        #if an X amount of time passed since last valid node and there is a minimum move, a new valid node is created
        if node.weight != 0:
            if self.time_since_last_valid_node >= self.valid_node_every and \
                np.sum((node.position - self.memory.last_valid_node.position)**2) >= self.min_distance_for_valid_node:
                node.weight = 0
                self.time_since_last_valid_node = 0
            else:
                node.weight = -1
            
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"Clustering node: {node.id} with weight {node.weight} and sequence {node.sequence_id} and position {node.position.tolist()}\n")

        #if node is not valid
        if node.weight != 0:
            node.region_id = self.memory.last_valid_node.region_id
            self.memory.nodes[node.id] = node
            self.memory.last_valid_node.not_valid_successive_nodes.append(node.id)
            
        #valid node
        else:
            links_from_this_node = []
            self.memory.nodes[node.id] = node
            
            #if same sequence, linked to the previous valid node
            
            if node.sequence_id == self.memory.last_valid_node.sequence_id:
                links_from_this_node.append(Link(node.id, self.memory.last_valid_node.id, 0))

            #KDTree to search nearest nodes IN THIS SEQUENCE
            #if a minimum amount of time is passed and the node is not the last valid
            candidates_nodes = list(filter(lambda x: x.weight == 0 and \
                abs(node.timestamp - x.timestamp) > (self.min_time_for_link * self.time_multiplier) and \
                    x.id != self.memory.last_valid_node.id, self.nodes[:self.current_node_idx]))
            pos = np.asarray(list(map(lambda x: x.position, candidates_nodes)))
            if len(pos) > 0:
                kd_tree = KDTree(pos)
                indices = kd_tree.query_ball_point(node.position,
                                                    self.max_distance_for_link, #search in a radius
                                                    workers=4,
                                                    return_sorted=True)
                links_counter = 0
                for idx in indices:
                    if links_counter >= self.max_links_intra_sequence: #max links to add intra sequence
                        break
                    #if a minimum amount of time is passed and the node is not the last valid, a new link is added
                    links_from_this_node.append(Link(node.id, candidates_nodes[idx].id, 0))
                    if globals.debug:
                        with open(globals.clustering_file, "a") as f:
                            f.write(f"Max distance: {self.max_distance_for_link}\n")
                            f.write(f"New link intra sequence: {node.id}->{candidates_nodes[idx].id}\n")
                            f.write(f"Distance: {np.sum((node.position-candidates_nodes[idx].position)**2)}\n")
                    links_counter += 1
            
            if self.kdtree is not None:        
                self._connect_with_prev_sequences(node, links_from_this_node)

            self._update_before_clustering(node, links_from_this_node)

            #clustering
            self.clustering(node, dynamic)
            
        self.current_node_idx += 1
        return True
        
__all__ = ["ClusteringFromGroundtruth"]