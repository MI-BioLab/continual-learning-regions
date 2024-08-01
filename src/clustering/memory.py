import numpy as np
import math
import time

from .region import Region
from . import constants

import globals

"""
Memory class which represents the map clustered so far and embedds methods to perform dyanmic clustering.
"""
class Memory:
    def __init__(self, desired_average_cardinality, mesh_shape_factor, radius_upper_bound):
        self.nodes = dict() #signatures in STM + WM
        self.last_valid_node = None
        self.total_connections = 0
        self.total_mesh = 0
        self.default_scattering = 0
        self.desired_average_cardinality = desired_average_cardinality
        self.mesh_shape_factor = mesh_shape_factor
        self.radius_upper_bound = radius_upper_bound
        self.threshold = 0
        self.scattering1_const = self.desired_average_cardinality * math.sqrt(self.desired_average_cardinality)
    
    """
    Return the nodes in the map sorted by id.
    """
    def get_nodes_sorted_by_id(self):
        nodes = []
        for id in sorted(list(self.nodes.keys())):
            if self.nodes[id].weight == 0:
                nodes.append(self.nodes[id])
        return nodes
        
    """
    Return the nodes for a given region.
    """
    def get_nodes_for_region(self, region_id):
        nodes = []
        for s in self.nodes.values():
            if s.region_id == region_id and s.weight == 0:
                nodes.append(s)
        return np.array(nodes)
        
    
    """
    Return the Region given its id.
    """
    def get_region(self, region_id):
        start_time = time.time()                
        nodes = self.get_nodes_for_region(region_id)
        to_return = Region.compute(nodes, self.default_scattering)
        end_time = time.time()  
        with open(globals.clustering_file, "a") as f:
            f.write(f"Get region time: {end_time - start_time}\n")
        return to_return
    
    """
    Method to update clustering params.
    """
    def update_global_clustering_params(self):
        self.update_default_scattering()
        self.update_threshold()
    
    """
    Default scattering formula.
    """
    def update_default_scattering(self):
        self.default_scattering = self.total_mesh * self.mesh_shape_factor / constants.K_2_PI

    """
    Clustering threshold formula.
    """
    def update_threshold(self):
        self.threshold = self.default_scattering * self.scattering1_const / 2

    """
    Utility method to traverse the graph.
    """
    def traverse(self, current_node, nodes_signed):
        if current_node.id not in nodes_signed:
            nodes_signed.add(current_node.id)
            for id_to in current_node.links.keys(): 
                if self.nodes[id_to].region_id == current_node.region_id and id_to not in nodes_signed:
                    self.traverse(self.nodes[id_to], nodes_signed)

    """
    Method which checks if a node is reassignable to another region.
    """
    def is_removable(self, node):
        n_nodes_for_region = len(self.get_nodes_for_region(node.region_id))
        if n_nodes_for_region == 1:
            return False
        initial_region_id = node.region_id
        nodes_signed = set()
        node_connected = None
        for id_to in node.links.keys(): #take the first signature connected that is of the same region
            if self.nodes[id_to].region_id == node.region_id:
                node_connected = self.nodes[id_to]
                break
        node.region_id = -1 #change temporarily region to avoid the traverse of this node
        if node_connected is not None: #it should never be none unless a region has only one node
            #start from this node, it has to traverse the entire region, excluding the initial signature and 
            #sign all the nodes (so that they are still connected even if the signature is removed)
            #so the connectivity constraint is maintained
            self.traverse(node_connected, nodes_signed)
        removable = len(nodes_signed) == (n_nodes_for_region - 1) #current was temporarily removed
        node.region_id = initial_region_id #restored to initial
        return removable

    """
    Move method of the scattering-based clustering.
    """
    def move(self, node, just_moved):
        if node.id in just_moved:
            return
        if self.is_removable(node):
            initial_region = self.get_region(node.region_id)
            
            min_delta_scattering = 1e10
            candidate = None
            
            regions_visited = set()
            for id_to in node.links.keys(): #links from this node
                if self.nodes[id_to].region_id != initial_region.id: 
                    if self.nodes[id_to].region_id in regions_visited:
                        continue
                    
                    regions_visited.add(self.nodes[id_to].region_id)
                    region = self.get_region(self.nodes[id_to].region_id)
                    
                    node.region_id = region.id #temporarly changed
                    updated_region = self.get_region(region.id)
                    node.region_id = initial_region.id #restored to initial
                    
                    delta_scattering = updated_region.scattering2 - region.scattering2
                    
                    if delta_scattering < min_delta_scattering and np.sum((updated_region.centroid - node.position)**2) < (self.radius_upper_bound**2):
                        min_delta_scattering = delta_scattering
                        candidate = updated_region
                    
            if candidate is not None:
                node.region_id = candidate.id
                #if true move v to candidate (already done), if not restore initial condition
                if self.get_region(initial_region.id).scattering2 - initial_region.scattering2 + min_delta_scattering < 0:
                    for id in node.not_valid_successive_nodes:
                        just_moved[id] = (self.nodes[id].region_id, node.region_id)
                        self.nodes[id].region_id = node.region_id
                    just_moved[node.id] = (initial_region.id, node.region_id)
                    for id_to in node.links.keys():
                        if self.nodes[id_to].region_id == initial_region.id: #nodes from the same region that is connected to v
                            self.move(self.nodes[id_to], just_moved)
                else:
                    node.region_id = initial_region.id

            else:
                node.region_id = initial_region.id
                
                
__all__ = ["Memory"]
        
    