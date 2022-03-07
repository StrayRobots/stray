import unittest
import os
import numpy as np
from stray.scene import Scene, NotASceneException
from stray.renderer import Renderer
import shutil

class TestScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene_path = os.path.join('.', 'tests', 'fixtures', 'bottle')
        cls.scene = Scene(scene_path)
        cls.renderer = Renderer(cls.scene)

    def test_open(self):
        self.assertIsNotNone(self.scene.mesh)
        self.assertEqual(len(self.scene.bounding_boxes), 1)

    def test_split(self):
        objects = self.scene.objects()
        bg = self.scene.background()
        self.assertEqual(len(objects), 1)
        self.assertLess(objects[0].vertices.shape[0], bg.vertices.shape[0])

    def test_trajectory(self):
        self.assertEqual(len(self.scene.poses), 1048)
        np.testing.assert_equal(self.scene.poses[0], np.eye(4))
        self.assertGreater(np.linalg.norm(self.scene.poses[1] - np.eye(4), 1), 0.0)

    def test_render(self):
        depth = self.renderer.render_depth(0)
        depth2 = self.renderer.render_depth(1)
        self.assertEqual(depth.shape[0], 480)
        self.assertEqual(depth.shape[1], 640)
        self.assertTrue(np.linalg.norm(depth - depth2, 1) > 0.0)

    def test_render_segmentation(self):
        seg = self.renderer.render_segmentation(0, colors=False)
        self.assertEqual(seg.shape, (480, 640))
        self.assertEqual(seg.max(), 1)
        self.assertLess(seg.mean(), 0.1)

        seg = self.renderer.render_segmentation(10, colors=True)
        self.assertEqual(seg.shape, (480, 640, 3))
        colors = np.unique(seg.reshape(-1, 3), axis=0)
        self.assertEqual(colors.shape[0], 2)


class TestValidateScene(unittest.TestCase):
    def test_not_scene(self):
        with self.assertRaises(NotASceneException):
            Scene.validate_path("/tmp")

        with self.assertRaises(NotASceneException):
            Scene.validate_path("/tmp/does_not_exist")

    def test_actual_scene(self):
        random_scene_path = '/tmp/scene_hash123'
        try:
            os.makedirs(random_scene_path, exist_ok=True)
            with open(os.path.join(random_scene_path, 'camera_intrinsics.json'), 'wt') as f:
                f.write("\n")
            path = Scene.validate_path(random_scene_path)
            self.assertEqual(path, random_scene_path)

            os.makedirs(os.path.join(random_scene_path, 'color'), exist_ok=True)
            path = Scene.validate_path(random_scene_path)
            self.assertEqual(path, random_scene_path)

            corrected = Scene.validate_path(os.path.join(random_scene_path, 'rgb.mp4'))
            self.assertEqual(corrected, random_scene_path)

        finally:
            shutil.rmtree(random_scene_path)



if __name__ == "__main__":
    unittest.main()

