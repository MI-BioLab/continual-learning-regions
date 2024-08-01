from scipy.spatial.transform import Rotation as R
from .base_dataset import BaseDataset

class TUMDataset(BaseDataset):
    """
    Pose Ground-Truth reader that conforms to the TUM RGB-D dataset format
    https://vision.in.tum.de/data/datasets/rgbd-dataset
    """
    
    # TUM represents odometry as a 3x1 vector tx, ty, tz
    # and a quaternion qx, qy, qz, qw.
    def __init__(self):
        self.COL_NAMES = self.TUM_LIKE
        self.ROT_COL_NAMES = self.ROT_TUM_LIKE
        self.n_rows_to_skip = 3
        
    def get_timestamps(self, poses): 
        return poses["time"].to_numpy(dtype=float)

    def get_angles(self, poses, rotation_axis):
        return self._get_angles_TUM_like(poses, rotation_axis)

__all__ = ["TUMDataset"]