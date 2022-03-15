import numpy as np

def compute_camera_matrix(fov, image_width, image_height):
    cx = image_width * 0.5
    cy = image_height * 0.5
    f = image_height / np.tan(np.deg2rad(fov) * 0.5) * 0.5
    return np.array([[f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]])

