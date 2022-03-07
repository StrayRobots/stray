import unittest
import numpy as np
from stray import linalg
from scipy.spatial.transform import Rotation

class LinalgTest(unittest.TestCase):
    def test_transform_points(self):
        points = np.random.randn(10, 3)
        points[2, :] = np.zeros(3)
        T = np.eye(4)
        transformed = linalg.transform_points(T, points)
        np.testing.assert_equal(transformed, points)

        T[:3, :3] = Rotation.random().as_matrix()
        T[:3, 3] = np.random.randn(3)
        transformed = linalg.transform_points(T, points)
        out = points.copy()
        for i in range(points.shape[0]):
            v = points[i, :]
            out[i] = T[:3, :3] @ points[i] + T[:3, 3]
        np.testing.assert_allclose(transformed, out)


if __name__ == "__main__":
    unittest.main()
