from typing import Sequence

from .base_clustering import BaseClustering

"""
Class to perform clustering given a graph (created by a SLAM algorithm such as RTABMap).
Nodes with relative poses and links intra-sequences must be provided.
"""
class ClusteringFromGraph(BaseClustering):
    def __init__(self, 
                 desired_average_cardinality, 
                 mesh_shape_factor, 
                 radius_upper_bound,
                 max_distance_for_link,
                 max_links,
                 link_each_sequence,
                 restart_after_each_sequence,
                 current_region_id=0):
        super().__init__(desired_average_cardinality, 
                         mesh_shape_factor, 
                         radius_upper_bound, 
                         max_distance_for_link,
                         max_links,
                         link_each_sequence,
                         restart_after_each_sequence,
                         current_region_id)
        
    """
    Set the current sequence to cluster.
    """
    def set_current_sequence(self, 
                             nodes: Sequence, 
                             links: Sequence = None):  
        super().set_current_sequence(nodes, links)
        self.link_between_valid_nodes = [link for link in links if all(node.weight == 0 for node in nodes if node.id in (link.id_from, link.id_to))]
    
    """
    Function to clusterize the next node in the sequence.
    """
    def clusterize_next_node(self, dynamic=True):
        if self.current_node_idx >= len(self.nodes):
            return False
        
        super().clusterize_next_node(dynamic)
        node = self.nodes[self.current_node_idx]
        
        if self.current_node_idx == 0: #new sequence
            if node.weight != 0: #first node of a sequence always set as valid
                node.weight = 0
            if len(self.memory.nodes) == 0: #first sequence
                node.region_id = self.current_region_id
                self.memory.nodes[node.id] = node
                self.memory.last_valid_node = node 
                self.current_node_idx += 1
                return True    
                
        
        if node.weight != 0: #node not valid
            node.region_id = self.memory.last_valid_node.region_id
            self.memory.nodes[node.id] = node
            self.memory.last_valid_node.not_valid_successive_nodes.append(node.id)   
            
        else:                
            self.memory.nodes[node.id] = node
                                    
            links_from_this_node = [link for link in self.link_between_valid_nodes if link.id_from == node.id and link.id_to < node.id] #links from this node seen so far
            
            if self.kdtree is not None:        
                self._connect_with_prev_sequences(node, links_from_this_node)
            
            self._update_before_clustering(node, links_from_this_node)
            
            #clustering
            self.clustering(node, dynamic)
            
        self.current_node_idx += 1
        return True
        
__all__ = ["ClusteringFromGraph"]
