import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

cube_vertices = np.array([
    [-1., -1., -1.],
    [-1., -1.,  1.],
    [-1.,  1., -1.],
    [-1.,  1.,  1.],
    [ 1., -1., -1.],
    [ 1., -1.,  1.],
    [ 1.,  1., -1.],
    [ 1.,  1.,  1.]], dtype=np.float32) * 0.5

cube_indices = np.array([
    [0., 0., 0.],
    [0., 0.,  1.],
    [0.,  1., 0.],
    [0.,  1.,  1.],
    [ 1., 0., 0.],
    [ 1., 0.,  1.],
    [ 1.,  1., 0.],
    [ 1.,  1.,  1.]], dtype=np.float32)

rectangle_vertices = np.array([
    [-1.0, -1.0, 0.0],
    [-1.0,  1.0, 0.0],
    [ 1.0, -1.0, 0.0],
    [ 1.0,  1.0, 0.0]]) * 0.5

class Primitive:
    def keypoints(self):
        """
        Points that should be detected by a keypoint detector and visualized as keypooints.
        """
        raise NotImplementedError()

class BoundingBox:
    """An abstraction of a bounding box."""

    def __init__(self, data):
        self.position = np.array(data['position'])
        self.dimensions = np.array(data['dimensions'])
        q = data['orientation']
        self.orientation = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']])
        self.instance_id = data.get('instance_id', 0)

    def cut(self, mesh):
        """
        Cuts the background out of the mesh, removing anything outside the bounding box.
        mesh: trimesh mesh
        returns: trimesh mesh for object
        """
        object_mesh = mesh
        axes = np.eye(3)
        for direction in [-1.0, 1.0]:
            for i, axis in enumerate(axes):
                normal = direction * self.orientation.apply(axis * 0.5)
                origin = self.position + normal * self.dimensions[i]
                object_mesh = trimesh.intersections.slice_mesh_plane(object_mesh, -normal, origin)
        return object_mesh

    def cut_pointcloud(self, pointcloud):
        """
        Returns the points which are inside the bounding box.
        pointcloud: N x 3 np.array
        returns: P x 3 points inside the bounding box
        """
        axes = np.eye(3)
        points_local = self.orientation.inv().apply(pointcloud - self.position)
        x_size = self.dimensions[0] * 0.5
        y_size = self.dimensions[1] * 0.5
        z_size = self.dimensions[2] * 0.5
        inside_x = np.bitwise_and(points_local[:, 0] < x_size, points_local[:, 0] > -x_size)
        inside_y = np.bitwise_and(points_local[:, 1] < y_size, points_local[:, 1] > -y_size)
        inside_z = np.bitwise_and(points_local[:, 2] < z_size, points_local[:, 2] > -z_size)
        mask = np.bitwise_and(np.bitwise_and(inside_x, inside_y), inside_z)
        return pointcloud[mask]

    def background(self, mesh):
        """
        Cuts the object out of the mesh, removing everything inside the bounding box.
        mesh: trimesh mesh
        returns: trimesh mesh for background
        """
        axes = np.eye(3)
        background = trimesh.Trimesh()
        for direction in [-1.0, 1.0]:
            for i, axis in enumerate(axes):
                normal = direction * self.orientation.apply(axis * 0.5)
                origin = self.position + normal * self.dimensions[i]
                outside = trimesh.intersections.slice_mesh_plane(mesh, normal, origin)
                background = trimesh.util.concatenate(background, outside)
        return background

    def vertices(self):
        vertices = cube_vertices * self.dimensions
        return self.position + self.orientation.apply(vertices)


class Keypoint:
    def __init__(self, data):
        self.class_id = data.get('instance_id', 0)
        self.position = data['position']

    def keypoints(self):
        return [self.position]

class Rectangle:
    def __init__(self, data):
        self.center = np.array(data['center'])
        self.class_id = data['class_id']
        q = data['orientation']
        self.orientation = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']])
        self.size = np.array(data['size'] + [0.0])

    def keypoints(self):
        vertices = rectangle_vertices * self.size
        return self.center + self.orientation.apply(vertices)

