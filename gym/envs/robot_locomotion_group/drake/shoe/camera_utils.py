import numpy as np
import math

from gym.envs.robot_locomotion_group.drake.transform_utils import transform_from_dict

def camera_transform_from_dict(data):
    return transform_from_dict(data)

def camera_K_matrix_from_dict(data):
    width = data['width']
    height = data['height']
    fov_y = data['fov_y']

    cx = width/2.
    cy = height/2.
    fy = height/2. * (1 / math.tan(fov_y/2.))
    fx = fy

    K = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
    return K

def remove_depth_out_of_range(depth_raw, dtype=None):
    """
    Sets any depth reading beyond max range or below min range to zero.
    Optionally casts result to datatype `dtype`. If dtype=None, no casting done.
    """
    if dtype is None:
        dtype = depth_raw.dtype

    depth = np.copy(depth_raw)
    if depth_raw.dtype == np.uint16:
        max_val = np.iinfo(np.uint16).max # max value of uint16
    elif depth_raw.dtype == np.float32:
        max_val = np.inf
    else:
        raise TypeError("Depth image with data type %s not recognized." % str(depth_raw.dtype))
    depth[depth == max_val] = 0

    max_measure = np.max(depth)
    if np.issubdtype(dtype, np.integer):
        max_express = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        max_express = np.finfo(dtype).max
    else:
        raise TypeError("Specified dtype %s not recognized." % str(dtype))
    if max_measure > max_express:
        raise OverflowError("np.max(depth)=%.1f > dtype.max=%.1f" % (max_measure, max_express))

    return depth.astype(dtype)
