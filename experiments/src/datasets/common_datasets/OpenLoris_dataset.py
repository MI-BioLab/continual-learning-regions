import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from .base_dataset import BaseDataset

class OpenLorisDataset(BaseDataset):
    """
    Pose Ground-Truth reader that conforms to the OpenLoris-Scene dataset format
    https://lifelong-robotic-vision.github.io/dataset/scene.html
    """
    
    # OpenLoris represents odometry as a 3x1 vector tx, ty, tz
    # and a quaternion qx, qy, qz, qw as the TUM dataset.
    def __init__(self):
        self.COL_NAMES = self.TUM_LIKE
        self.ROT_COL_NAMES = self.ROT_TUM_LIKE
        self.n_rows_to_skip = 1
        
    def get_timestamps(self, poses): 
        return poses["time"].to_numpy(dtype=float)
    
    def get_2D_translations(self, poses):
        return self.get_translations(poses, ["tx", "ty"])

    def get_angles(self, poses, rotation_axis):
        return self._get_angles_TUM_like(poses, rotation_axis)

__all__ = ["OpenLorisDataset"]