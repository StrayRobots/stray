import os
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from stray import camera

class NotASceneException(ValueError):
    pass

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
    """An abstraction of a  keypoint."""

    def __init__(self, data):
        self.instance_id = data.get('instance_id', 0)
        self.position = data['position']

class Scene:
    """An abstraction of a scanned scene."""

    def __init__(self, path):
        self.scene_path = path
        self._read_mesh()
        self._read_annotations()
        self._read_metadata()
        self._bounding_boxes = None
        self._keypoints = None
        self._poses = None
        self._metadata = None
        self._read_intrinsics()

    def _read_mesh(self):
        mesh_file = os.path.join(self.scene_path, 'scene', 'integrated.ply')
        if os.path.exists(mesh_file):
            self.mesh = trimesh.load(mesh_file)
        else:
            self.mesh = None

    def _read_point_cloud(self):
        point_cloud_file = os.path.join(self.scene_path, 'scene', 'cloud.ply')
        if os.path.exists(point_cloud_file):
            self.point_cloud = trimesh.load(point_cloud_file)
        else:
            self.point_cloud = None

    def _read_annotations(self):
        annotation_file = os.path.join(self.scene_path, 'annotations.json')
        if not os.path.exists(annotation_file):
            self.annotations = {}
        else:
            with open(annotation_file, 'rt') as f:
                self.annotations = json.load(f)

    def _read_metadata(self):
        metadata_path = os.path.join(os.path.dirname(self.scene_path.rstrip("/")), "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rt') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _read_trajectory(self):
        self._poses = []
        if os.path.exists(os.path.join(self.scene_path, 'scene', 'trajectory.log')):
            with open(os.path.join(self.scene_path, 'scene', 'trajectory.log'), 'rt') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 5):
                    rows = [np.fromstring(l, count=4, sep=' ') for l in lines[i+1:i+5]]
                    self._poses.append(np.stack(rows))

    def _read_intrinsics(self):
        if os.path.exists(os.path.join(self.scene_path, 'camera_intrinsics.json')):
            with open(os.path.join(self.scene_path, 'camera_intrinsics.json')) as f:
                camera_data = json.load(f)
            self.camera_matrix = np.array(camera_data['intrinsic_matrix']).reshape(3, 3).T
            self.frame_width = camera_data['width']
            self.frame_height = camera_data['height']

    def _process_annotations(self):
        self._bounding_boxes = []
        for bbox in self.annotations.get('bounding_boxes', []):
            self._bounding_boxes.append(BoundingBox(bbox))
        self._keypoints = []
        for keypoint in self.annotations.get('keypoints', []):
            self._keypoints.append(Keypoint(keypoint))

    def __len__(self):
        return len(self.get_image_filepaths())

    def camera(self):
        return camera.Camera((self.frame_width, self.frame_height), self.camera_matrix, np.zeros(4))


    def add_keypoint_annotation(self, keypoint: Keypoint):
        annotation_file = os.path.join(self.scene_path, 'annotations.json')
        if not os.path.exists(annotation_file):
            annotations = {}
        else:
            with open(annotation_file, 'rt') as f:
                annotations = json.load(f)
        if "keypoints" in annotations.keys():
            annotations["keypoints"].append(keypoint.to_dict())
        else:
            annotations["keypoints"] = [keypoint.to_dict()]

        json_object = json.dumps(annotations, indent = 4)

        with open(annotation_file, "w") as outfile:
            outfile.write(json_object)

        self._read_annotations()
        self._process_annotations()

    @property
    def poses(self):
        if self._poses is None:
            self._read_trajectory()
        return self._poses

    @property
    def bounding_boxes(self):
        if self._bounding_boxes is None:
            self._process_annotations()
        return self._bounding_boxes

    @property
    def keypoints(self):
        if self._keypoints is None:
            self._process_annotations()
        return self._keypoints

    @property
    def bbox_categories(self):
        categories = []
        for b in self.bounding_boxes:
            if not b.instance_id in categories:
                categories.append(b.instance_id)
        return categories

    @property
    def keypoint_categories(self):
        categories = []
        for k in self.keypoints:
            if not k.instance_id in categories:
                categories.append(k.instance_id)
        return categories

    @property
    def objects(self):
        """
        Returns a trimesh for each bounding box in the scene.
        returns: list[trimesh.Mesh]
        """
        objects = []
        for bbox in self.bounding_boxes:
            object_mesh = bbox.cut(self.mesh)
            objects.append(object_mesh)
        return objects

    def get_image_filepaths(self):
        paths = os.listdir(os.path.join(self.scene_path, 'color'))
        paths = [path for path in paths if path.lower().split(".")[-1] in ['png', 'jpg', 'jpeg']]
        paths.sort()
        return list(map(lambda p: os.path.join(self.scene_path, 'color', p), paths))

    def image_size(self):
        images = self.get_image_filepaths()
        return Image.open(images[0]).size

    def get_depth_filepaths(self):
        paths = os.listdir(os.path.join(self.scene_path, 'depth'))
        paths = [path for path in paths if path.lower().split(".")[-1] == 'png']
        paths.sort()
        return list(map(lambda p: os.path.join(self.scene_path, 'depth', p), paths))

    def background(self):
        background = self.mesh
        for bbox in self.bounding_boxes:
            background = bbox.background(self.mesh)
        return background

    @staticmethod
    def validate_path(scene_path) -> str:
        """
        Checks if a path is an actual path. Returns a fixed path, if for example the path
        refers to a subfile or directory in the scene folder. If scene_path is legit, this is an identity function.

        throws NotASceneException if this doesn't look to be a scene folder.

        scene_path: str path to a potential path
        returns: str scene_path or fixed scene_path
        """
        def looks_like_scene(path):
            is_dir = path.is_dir()
            has_color_subdir = (path / "color").is_dir()
            has_scene_subdir = (path / "scene").is_dir()
            has_intrinsics = (path / "camera_intrinsics.json").is_file()
            return is_dir and (has_color_subdir or has_scene_subdir or has_intrinsics)

        path = Path(scene_path)
        if looks_like_scene(path):
            return scene_path
        elif looks_like_scene(path.parent):
            return str(path.parent)
        elif looks_like_scene(path.parent.parent):
            return str(path.parent.parent)
        raise NotASceneException(f"The path {scene_path}")
