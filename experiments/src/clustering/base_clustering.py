from typing import Sequence

import numpy as np
from scipy.spatial import KDTree
import time

from .memory import Memory
from .node import Node
from .link import Link

import globals

"""
    Base class for clustering.    
"""
class BaseClustering:
    def __init__(self, 
                 desired_average_cardinality, 
                 mesh_shape_factor, 
                 radius_upper_bound,
                 max_distance_for_link,
                 max_links,
                 link_each_sequence,
                 restart_after_each_sequence,
                 current_region_id=0):
        self.desired_average_cardinality = desired_average_cardinality
        self.mesh_shape_factor = mesh_shape_factor
        self.radius_upper_bound = radius_upper_bound
        self.memory = Memory(desired_average_cardinality, mesh_shape_factor, radius_upper_bound)
        self.max_distance_for_link = max_distance_for_link if max_distance_for_link >= 0 else None  
        self.max_links = max_links if max_links >= 0 else None
        self.link_each_sequence = link_each_sequence
        self.restart_after_each_sequence = restart_after_each_sequence
        self.current_region_id = current_region_id
        self.just_moved = dict()
        self.kdtree = None
        self.sorted_valid_nodes = None
        self.nodes = []
        self.current_node_idx = 0    
        self.current_exp = 0    
        
    def _connect_with_prev_sequences(self,
                                     node,
                                     links):
        assert self.kdtree is not None, "kdtree should not be None"
        start_time = time.time()            
        indices = self.kdtree.query_ball_point(node.position, 
                                          self.max_distance_for_link,
                                          workers=4,
                                          return_sorted=True)
 
        end_time = time.time()
        with open(globals.clustering_file, "a") as f:
            f.write(f"Query ball point time: {end_time - start_time}\n")
        
        if self.link_each_sequence:
            sequence_linked = dict()
        else:
            links_counter = 0
        for idx in indices:
            if self.link_each_sequence:
                if node.sequence_id not in sequence_linked:
                    sequence_linked[node.sequence_id] = 0
                if sequence_linked[node.sequence_id] >= self.max_links and self.max_links >= 0:
                    continue
                else:
                    sequence_linked[node.sequence_id] += 1
            else:
                if links_counter >= self.max_links and self.max_links >= 0:
                    break
                else:
                    links_counter += 1
            links.append(Link(node.id, self.sorted_valid_nodes[idx].id, 0))
            
    def _update_before_clustering(self, node, links):
        new_gaps = 0
        for l in links:
            node.links[l.id_to] = l
            new_gaps += np.sqrt(np.sum((node.position - self.memory.nodes[l.id_to].position)**2)) #*2 ? To consider from and to both?
            self.memory.nodes[l.id_to].links[l.id_from] = Link(l.id_to, l.id_from, 0)
    
        #update clustering parameters
        self.memory.total_mesh = (self.memory.total_mesh * self.memory.total_connections + new_gaps) / (self.memory.total_connections + len(node.links))
        self.memory.total_connections = self.memory.total_connections + len(node.links)
        self.memory.update_global_clustering_params()
    
    def set_current_sequence(self, 
                             nodes: Sequence, 
                             links: Sequence = None):
        self.nodes = nodes
        self.current_node_idx = 0
        if self.restart_after_each_sequence:
            self.memory = Memory(self.desired_average_cardinality, self.mesh_shape_factor, self.radius_upper_bound)
            self.kdtree = None
        
    def clusterize_next_node(self, dynamic=True):
        self.just_moved = dict()
        
    def clusterize_next_experience(self, experience_size, dynamic=True):
        self.experience_moved = dict()
        for _ in range(experience_size):
            cont = self.clusterize_next_node()
            for k, v in self.just_moved.items():
                self.experience_moved[int(k)] = (int(v[0]), int(v[1]))                
            if not cont:
                break
        self.update_when_sequence_ended()
        self.current_exp += 1
        with open(globals.clustering_file, "a") as f:
            f.write(f"Nodes moved in this experience: {self.experience_moved}\n")
        return cont
        
    def align_sequence_until_now(self, nodes):
        assert self.kdtree is not None
        ids_added = []
        for i in range(len(nodes)):
            node = nodes[i]
            
            if self.kdtree is not None:
                #it always check to add links to nn in other sequences (kdtree is update only at the end of a sequence)
                distances, indices = self.kdtree.query(node.position, 
                                                        k=1, 
                                                        workers=4
                                                        )

                if distances < 20:
                    if len(ids_added) <= 0 or abs(ids_added[-1] - node.id) < 300:
                        node.region_id = self.sorted_valid_nodes[indices].region_id
                        ids_added.append(node.id)
            
        with open(globals.clustering_file, "a") as f:
            f.write(f"Nodes aligned: {len(ids_added)} / {len(nodes)}\n")
        
    def align_sequence(self):
        assert self.kdtree is not None
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            
            if self.kdtree is not None:
                #it always check to add links to nn in other sequences (kdtree is update only at the end of a sequence)
                distances, indices = self.kdtree.query(node.position, 
                                                        k=1, 
                                                        workers=4
                                                        )
            
                if distances != np.inf:
                    node.region_id = self.sorted_valid_nodes[indices].region_id
        
    def update_when_sequence_ended(self):
        start_time = time.time()
        self.sorted_valid_nodes = self.memory.get_nodes_sorted_by_id()
        valid_positions = np.asarray(list(map(lambda n: n.position, self.sorted_valid_nodes)))
        self.kdtree = KDTree(valid_positions)
        end_time = time.time()
        if globals.debug:
            with open(globals.clustering_file, "a") as f:
                f.write(f"KDTree creation time: {end_time - start_time}\n")
        
    def clustering(self, node: Node, dynamic=True):
        region_candidate = None
        min_scattering = 1e10
        regions_visited = set()       
        for id_to in node.links.keys(): 
            if self.memory.nodes[id_to].region_id in regions_visited:
                continue
            if globals.debug:
                with open(globals.clustering_file, "a") as f:
                    f.write(f"Links id to {id_to}\n")
            regions_visited.add(self.memory.nodes[id_to].region_id)
            region = self.memory.get_region(self.memory.nodes[id_to].region_id) 
            node.region_id = region.id #temporarly changed
            updated_region = self.memory.get_region(node.region_id)
            node.region_id = -1 #restored to initial
            if globals.debug:
                with open(globals.clustering_file, "a") as f:
                    f.write(f"{updated_region.scattering2 - region.scattering2} < {self.memory.threshold + self.memory.default_scattering} " \
                        f"and {np.sum((updated_region.centroid - node.position)**2)} < {(self.radius_upper_bound**2)} " \
                            f"and {updated_region.scattering2} < {min_scattering}\n")
            if updated_region.scattering2 - region.scattering2 < self.memory.threshold + self.memory.default_scattering and \
                np.sum((updated_region.centroid - node.position)**2) < (self.radius_upper_bound**2) \
                    and updated_region.scattering2 < min_scattering: 
                    min_scattering = updated_region.scattering2
                    region_candidate = updated_region
                                    
                    
        #no region candidates
        if region_candidate is None:
            self.current_region_id += 1
            node.region_id = self.current_region_id
            if globals.debug:
                with open(globals.clustering_file, "a") as f:
                    f.write(f"Node {node.id}: new region with id {node.region_id}\n")
                    
        else:
            node.region_id = region_candidate.id
            if globals.debug:
                with open(globals.clustering_file, "a") as f:
                    f.write(f"Node {node.id}: assigned to region {node.region_id}\n")

        if dynamic:   
            start_time = time.time()
            self.memory.move(node, self.just_moved)
            for id_to in node.links.keys():
                self.memory.move(self.memory.nodes[id_to], self.just_moved)
            end_time = time.time()
            if globals.debug:
                with open(globals.clustering_file, "a") as f:
                    f.write(f"Move time: {end_time - start_time}\n")
                                    
        self.memory.last_valid_node = node
        
__all__ = ["BaseClustering"]