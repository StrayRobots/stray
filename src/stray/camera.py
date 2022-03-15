import cv2
import numpy as np
import open3d as o3d

class Camera:
    """An abstraction of a pihnole camera."""

    def __init__(self, size, K, D=np.zeros(4)):
        """
        K: 3 x 3 camera projection matrix.
        D: distortion parameters.
        
        Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value
        """
        self.size = size
        self.camera_matrix = K
        self.Kinv = np.linalg.inv(K)
        self.distortion = D

    def project(self, points, T_CW=np.eye(4)):
        """
        points: N x 3 3D points.
        T: optional transform matrix to convert points into camera coordinates.
        returns: N x 2 image coordinate points.
        Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value
        """
        R, _ = cv2.Rodrigues(T_CW[:3, :3])
        out, _ = cv2.projectPoints(points, R, T_CW[:3, 3], self.camera_matrix, self.distortion)
        return out[:, 0, :]

    def scale(self, new_size):
        """Summary line.

        Extended description of function.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value
        """

        scale_x = new_size[0] / self.size[0]
        scale_y = new_size[1] / self.size[1]
        new_K = get_scaled_camera_matrix(self.camera_matrix, scale_x, scale_y)
        return Camera(new_size, new_K, self.distortion)

    def unproject(self, xy, z):
        """
        xy: image points x and y coordinates. N x 2
        z: depth value, how far is the point. N sized array
        returns: 3D point xyz coordinates relative to camera frame.
        """
        xy = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)
        X = (self.Kinv @ xy[:, :, None])[:, :, 0] * z[:, None]
        return X

def scale_intrinsics(o3d_intrinsics, new_width, new_height):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value
    """

    old_height = o3d_intrinsics.height
    old_width = o3d_intrinsics.width
    if old_width == new_width and old_height == new_height:
        return o3d_intrinsics
    fx, fy = o3d_intrinsics.get_focal_length()
    cx, cy = o3d_intrinsics.get_principal_point()
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    fx, fy = fx * scale_x, fy * scale_y
    cx, cy = cx * scale_x, fy * scale_y
    return o3d.camera.PinholeCameraIntrinsic(int(new_width), int(new_height), fx, fy, cx, cy)

def get_scaled_camera_matrix(camera_matrix, width_scale, height_scale):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value
    """


    K = np.eye(3)
    K[0, 0] = camera_matrix[0, 0] * width_scale
    K[1, 1] = camera_matrix[1, 1] * height_scale
    K[0, 2] = camera_matrix[0, 2] * width_scale
    K[1, 2] = camera_matrix[1, 2] * height_scale
    return K
