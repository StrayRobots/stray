import open3d as o3d
import numpy as np

def transform(point, T):
    v = np.ones((4, 1))
    v[:3, 0] = point
    return (T @ v)[:3, 0]

class VisualDebugger:
    def __init__(self):
        self.geometries = []

    def add_point(self, point, T_WP=np.eye(4), color=np.array([0, 1, 0])):
        point_W = transform(point, T_WP)
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.01).paint_uniform_color(color)
        translation = np.eye(4)
        translation[:3, 3] = point_W
        sphere = sphere.transform(translation)
        self.geometries.append(sphere)

    def add_frame(self, frame, scale=0.1, color=None):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(scale, np.zeros(3))
        if color is not None:
            mesh.paint_uniform_color(color)
        self.geometries.append(mesh.transform(frame))

    def add_rgbd(self, color, depth, intrinsics, T_WC=np.eye(4)):
        depth = (depth * 1000.0).astype(np.uint16)
        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                depth_scale=1000.0, convert_rgb_to_intensity=False)

        camera_matrix = intrinsics['camera_matrix']
        pinhole = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'],
                camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole, extrinsic=np.linalg.inv(T_WC))
        self.geometries.append(point_cloud)

    def add_rectangle(self, width, height, T_WB=np.eye(4)):
        box = o3d.geometry.TriangleMesh.create_box(width, height, 0.001)
        box.paint_uniform_color(np.array([0.5, 0.5, 1.0]))
        self.geometries.append(box.transform(T_WB))

    def show(self):
        o3d.visualization.draw_geometries(self.geometries)

