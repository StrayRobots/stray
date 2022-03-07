import numpy as np
import pyrender
from matplotlib import cm

R_ogl_cv = np.array([[1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0]])
T_ogl_cv = np.eye(4)
T_ogl_cv[:3, :3] = R_ogl_cv
segmentation_colors = cm.tab10(np.linspace(0, 1, 10))[:, :3]

class Renderer:
    def __init__(self, scene):
        self.scene = scene
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.scene.frame_width,
                    viewport_height=self.scene.frame_height, point_size=1.0)
        self.camera = pyrender.IntrinsicsCamera(scene.camera_matrix[0, 0], scene.camera_matrix[1, 1],
                scene.camera_matrix[0, 2], scene.camera_matrix[1, 2])
        self._build_pyrender_scene()

    def _build_pyrender_scene(self):
        self.render_scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.camera_node = pyrender.Node(camera=self.camera, matrix=T_ogl_cv)
        self.render_scene.add(light)
        self.render_scene.add_node(self.camera_node)
        self.bg_node = pyrender.Node(pyrender.Mesh.from_trimesh(self.scene.background()))
        self.render_scene.add_node(self.bg_node)
        self.instance_nodes = []
    
    def add_scene_instance(self, instance):
        obj = instance.cut(self.scene.mesh)
        node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(obj))
        self.render_scene.add_node(node)
        self.instance_nodes.append(node)
    
    def clear_scene_instances(self):
        for node in self.instance_nodes:
            self.render_scene.remove_node(node)
        self.instance_nodes = []
    

    def update_camera_position(self, new_position):
        self.render_scene.remove_node(self.camera_node)
        self.camera_node = pyrender.Node(camera=self.camera, matrix=new_position)
        self.render_scene.add_node(self.camera_node)

    def render_depth(self, frame_index):
        camera_pose = self.scene.poses[frame_index]
        self.update_camera_position(T_ogl_cv @ camera_pose)
        return self.renderer.render(self.render_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    def render_segmentation(self, frame_index, colors=False):
        """
        Renders the semantic segmentation map for camera pose at index frame.
        If colors is set to true, will return a HxWx3 colored image instead of a HxW pixel to instance id matrix.
        """
        camera_pose = self.scene.poses[frame_index]
        self.update_camera_position(camera_pose @ T_ogl_cv)
        segmentation = {
            self.bg_node: (0.0, 0.0, 0.0),
        }
        for i, node in enumerate(self.instance_nodes):
            c = segmentation_colors[i]
            segmentation[node] = (c[0], c[1], c[2])
        seg, _ = self.renderer.render(self.render_scene, seg_node_map=segmentation, flags=pyrender.RenderFlags.SEG)
        seg = seg[:, :, 2]
        if not colors:
            return seg
        else:
            out = np.zeros((seg.shape[0], seg.shape[1], 3))
            out[seg == 0, :] = np.array([0, 0, 0])
            for bbox in self.scene.bounding_boxes:
                out[seg == (1 + bbox.instance_id), :] = segmentation_colors[bbox.instance_id]

            return (out * 255.0).round().astype(np.uint8)


