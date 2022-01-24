import numpy as np
import torch
import cv2
from stray.util.scene import Scene
from torch.utils.data import Dataset, ConcatDataset
from stray.util.heatmap_utils import get_heatmap
import pytorch_lightning as pl
import torch


I = np.eye(3)

def transform(T, vectors):
    return (T[:3, :3] @ vectors[:, :, None])[:, :, 0] + T[:3, 3]
class KeypointHeatmapScene(Dataset):
    def __init__(self, path, image_size, out_size, num_keypoints):
        self.scene_path = path
        self.scene = Scene(path)
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.out_width = out_size[0]
        self.out_height = out_size[1]
        self.num_instances = len(self.scene.keypoints)


        self.color_images = self.scene.get_image_filepaths()
        self.camera = self.scene.camera()

        self.map_camera = self.scene.camera().scale((self.out_width, self.out_height))

    def _get_cv_image(self, idx):
        image = cv2.imread(self.color_images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.scene)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        heatmaps = []

        T_CW = np.linalg.inv(self.scene.poses[idx])
        sorted_keypoint_positions = []
        for instance_id in [0,1,2,3]:
            for kp in self.scene.keypoints:
                if kp.instance_id == instance_id:
                    sorted_keypoint_positions.append(kp.position)

        keypoint_positions_W = np.array(sorted_keypoint_positions)
        keypoint_positions_C  = transform(T_CW, keypoint_positions_W)
        keypoint_positions_I = self.camera.project(keypoint_positions_C)

        cv_image = self._get_cv_image(idx)
        cv_image = cv2.resize(cv_image, (self.image_width, self.image_height))

        cv_image = cv_image.astype(np.float32)
        np_image = np.transpose(cv_image/255.0, [2, 0, 1])


        for point_2d_I, point_3d in zip(keypoint_positions_I, keypoint_positions_W):
            width_scale = self.out_width / self.image_width
            height_scale = self.out_height / self.image_height
            point_2d_I = [point_2d_I[0]*width_scale, point_2d_I[1]*height_scale]

            point_3d_C = transform(T_CW, point_3d[None])[0]

            diagonal_fraction = 1.5
            top_point = self.map_camera.project((point_3d_C  - I[1] * diagonal_fraction)[None])[0]
            bottom_point = self.map_camera.project((point_3d_C  + I[1] * diagonal_fraction)[None])[0]
            size = np.linalg.norm(top_point - bottom_point)
            lengthscale = np.sqrt(size**2/20.0)
            heatmap = get_heatmap(point_2d_I, self.out_width, self.out_height, lengthscale)
            heatmaps.append(heatmap)

        heatmaps = np.array(heatmaps)
        return torch.from_numpy(np_image).float(), torch.from_numpy(heatmaps).float()

class KeypointHeatmapDataset(ConcatDataset):
    def __init__(self, scene_paths, *args, **kwargs):
        scenes = []
        for scene_path in scene_paths:
            scenes.append(KeypointHeatmapScene(scene_path, *args, **kwargs))
        super().__init__(scenes)

class KeypointHeatmapDataModule(pl.LightningDataModule):
    def __init__(self, train_dirs, eval_dirs, train_batch_size, eval_batch_size, num_workers, num_keypoints, image_size, out_size):
        super().__init__()
        self.train_dirs = train_dirs
        self.eval_dirs = eval_dirs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.out_size = out_size
        self.num_keypoints = num_keypoints

    def train_dataloader(self):
        dataset = KeypointHeatmapDataset(self.train_dirs, self.image_size, self.out_size, self.num_keypoints)

        return torch.utils.data.DataLoader(dataset,
                    num_workers=self.num_workers,
                    batch_size=self.train_batch_size,
                    persistent_workers=True,
                    pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        dataset = KeypointHeatmapDataset(self.eval_dirs, self.image_size, self.out_size, self.num_keypoints)

        return torch.utils.data.DataLoader(dataset,
                    num_workers=self.num_workers,
                    batch_size=self.eval_batch_size,
                    persistent_workers=True,
                    pin_memory=torch.cuda.is_available())

