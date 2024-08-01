import numpy as np
from scipy.spatial import KDTree

import math

from . import constants

"""
Class which represent a Region. It keeps track of some clustering params.
"""
class Region:
    def __init__(self, 
                 id, 
                 cardinality, 
                 centroid,
                 mesh,
                 equivalent_radius,
                 scattering2
                 ):
        self.id = id
        self.centroid = centroid
        self.cardinality = cardinality
        self.mesh = mesh
        self.equivalent_radius = equivalent_radius
        self.scattering2 = scattering2
        
    def __eq__(self, __o: object) -> bool:
        self.id = __o.id
    
    @staticmethod
    def compute(signatures, default_scattering): 
        centroid = None
        cardinality = 0
        mesh = 0
        equivalent_radius = 0
        scattering2 = 0
        kdtree = None
        
        #cardinality
        cardinality = len(signatures)
        
        #centroid
        positions = np.array(list(map(lambda x: x.position, signatures)))
        centroid = np.mean(positions, 0)
            
        #gaps
        if cardinality > 1:
            gaps = []
            kdtree = KDTree(positions)
            for s in signatures:
                dd, ii = kdtree.query(s.position, k=[2]) 
                gaps.append(dd)
            gaps = np.array(gaps) 
            
            mesh = np.mean(gaps)
            equivalent_radius = constants.K * mesh * math.sqrt(cardinality)
            position_distances2 = 0
            for s in signatures:
                position_distances2 += np.sum((centroid - s.position)**2)
            scattering2 = position_distances2 / (equivalent_radius + 1e-7)
        elif cardinality == 1:
            scattering2 = default_scattering
            
        return Region(signatures[0].region_id,
                      cardinality,
                      centroid,
                      mesh,
                      equivalent_radius,
                      scattering2)
        
__all__ = ["Region"]