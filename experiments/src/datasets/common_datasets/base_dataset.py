import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


class BaseDataset():
    KITTI_LIKE = [
        'r00', 'r01', 'r02', 'tx', 'r10', 'r11', 'r12', 'ty', 'r20', 'r21',
        'r22', 'tz'
    ]

    ROT_KITTI_LIKE = [
        'r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22'
    ]

    TUM_LIKE = ['time', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']

    ROT_TUM_LIKE = ['qx', 'qy', 'qz', 'qw']
    
    ST_LUCIA = ['tx', 'ty']

    R_VEC = ["x", "y", "z"]

    COL_NAMES = None
    ROT_COL_NAMES = None
    n_rows_to_skip = 0
    translation_axis = ["tx", "ty", "tz"]
    sep = " "

    def read_file(self, input_file):
        poses = pd.read_csv(input_file, sep=self.sep, names=self.COL_NAMES)
        poses = poses.iloc[self.n_rows_to_skip:, :]
        return poses
    
    def get_timestamps(self, poses): 
        pass

    def get_translations(self, poses, translation_axis = None):
        """
        Returns the translations for the given axis.
        """
        if translation_axis is None:
            return poses[self.translation_axis].to_numpy(dtype=float)
        return poses[translation_axis].to_numpy(dtype=float)
    
    def get_2D_translations(self, poses):
        raise NotImplementedError

    def get_angles(self, poses, rotation_axis):
        pass

    def _get_angles_KITTI_like(self, poses, rotation_axis):
        """
        Returns the angles as Euler angles for the given axis.
        """
        rotations = poses[self.ROT_COL_NAMES].to_numpy().reshape(
            len(poses), 3, 3)
        rotations = R.from_matrix(rotations)
        rotations = rotations.as_euler('xyz', degrees=True)
        rotations = pd.DataFrame(rotations, columns=self.R_VEC)
        return rotations[rotation_axis].to_numpy(dtype=float)

    def _get_angles_TUM_like(self, poses, rotation_axis):
        """
        Returns the angles as Euler angles for the given axis.
        """
        rotations = poses[self.ROT_COL_NAMES].to_numpy().reshape(len(poses), 4)
        rotations = R.from_quat(rotations)
        rotations = rotations.as_euler('xyz', degrees=True)
        rotations = pd.DataFrame(rotations, columns=self.R_VEC)
        return rotations[rotation_axis].to_numpy(dtype=float)


__all__ = ["BaseDataset"]
