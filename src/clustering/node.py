"""
Class which represents a node of the graph.
Initially, the region assigned is -1.
The weight parameter says if a node is valid or not (according to RTABMap).
"""
class Node:
    def __init__(self, id, sequence_id, images, weight, position, timestamp=None, region_id=-1):
        self.id = id
        self.sequence_id = sequence_id
        self.images = images
        self.weight = weight
        self.position = position
        self.timestamp = timestamp
        self.region_id = region_id
        self.links = dict() #id_to, Link
        self.not_valid_successive_nodes = []
        
    
    def __eq__(self, __o: object) -> bool:
        self.id = __o.id
        
    def __str__(self) -> str:
        return f"Node(id: {self.id}, sequence_id: {self.sequence_id}, weight: {self.weight}, region: {self.region_id}, timestamp: {self.timestamp})"
        
__all__ = ["Node"]