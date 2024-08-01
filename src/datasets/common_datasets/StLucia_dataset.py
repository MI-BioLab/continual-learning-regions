from scipy.spatial.transform import Rotation as R
from .base_dataset import BaseDataset

class StLuciaDataset(BaseDataset):
    """
    Pose Ground-Truth reader that conforms to the Santa Lucia dataset format
    """
    
    # Santa Lucia ground truth is obtained through GPS (latitude and longitude), so only tx and ty are avaliable. 
    def __init__(self):
        self.COL_NAMES = self.ST_LUCIA
        self.n_rows_to_skip = 0
        self.translation_axis = self.ST_LUCIA
        self.sep = ","
        
    def get_timestamps(self, poses): 
        raise ValueError("Santa Lucia doesn't have timestamps")

    def get_angles(self, poses, rotation_axis):
        raise ValueError("Santa Lucia does not have rotations!")

__all__ = ["OpenLorisDataset"]