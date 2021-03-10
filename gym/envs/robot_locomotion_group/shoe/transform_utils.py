import numpy as np

from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform, RollPitchYaw

def get_data_from_dict(data, possible_keys):
    for key in possible_keys:
        if key in data:
            return data[key], key
    raise ValueError("None of possible keys found: " + possible_keys)

def transform_from_dict(data):
    pos, _ = get_data_from_dict(data, ['pos', 'position', 'translation'])
    orientation, key = get_data_from_dict(data, ['quat', 'quaternion', 'rpy'])
    if key == "rpy":
        return RigidTransform(RollPitchYaw(orientation), pos)
    else:
        return RigidTransform(Quaternion(orientation), pos)

def transform_to_dict(X):
    return {"translation": X.translation(),
            "quaternion": X.rotation().ToQuaternion().wxyz(),
            "raw": X.GetAsMatrix4()}

def transform_to_vector(X, rpy=True):
    """
    If rpy True, returns RollPitchYaw vector followed by translation vector.
    Otherwise, returns Quaternion vector followed by translation vector.
    """
    if rpy:
        vector = np.empty(6)
        vector[:3] = RollPitchYaw(X.rotation()).vector()
    else:
        vector = np.empty(7)
        vector[:4] = X.rotation().ToQuaternion().wxyz()
    vector[-3:] = X.translation()
    return vector

def transform_points(X, points):
    """
    X is 3x3 and points is Nx2 or X is 4x4 and points is Nx3
    """
    if isinstance(X, RigidTransform):
        X = X.GetAsMatrix4()
    assert X.shape[1] == points.shape[1] + 1
    N = points.shape[0]
    points_homogenous = np.vstack((points.T, np.ones((1, N))))
    transformed_points = np.matmul(X[:3], points_homogenous)
    return transformed_points.T
