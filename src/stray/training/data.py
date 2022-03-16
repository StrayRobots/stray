import torch
import numpy as np
import cv2
import pytorch_lightning as pl
import albumentations as A
from stray.scene import Scene
from torch.utils.data import Dataset, ConcatDataset
from stray.training import transform, get_heatmap, I

class StrayKeypointScene(Dataset):
    def __init__(self, path, config, image_size, out_size, eval=False):
        self.scene_path = path
        self.scene = Scene(path)
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.out_width = out_size[0]
        self.out_height = out_size[1]
        self.primitive = config['primitive']
        self.num_instances = config['num_instances']

        self.eval = eval

        self.color_images = self.scene.get_image_filepaths()
        self.camera = self.scene.camera()

        self.map_camera = self.scene.camera().scale((self.out_width, self.out_height))

        self.transform = A.Compose([
            A.RandomResizedCrop(width=self.image_width, height=self.image_height, scale=(0.3, 1.0), ratio=(0.5, 1.3333333333333333), always_apply=True, p=1.0),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.keypoints_W = self._get_keypoints()

    def _get_keypoints(self):
        if self.primitive == "keypoints":
            world_points = np.zeros((self.num_instances, 3))
            for keypoint in self.scene.keypoints:
                world_points[keypoint.class_id] = keypoint.position
        elif self.primitive == "rectangle":
            world_points = np.zeros((4*self.num_instances, 3))
            for rectangle in self.scene.rectangles:
                keypoints = rectangle.keypoints()
                world_points[rectangle.class_id:rectangle.class_id+4] = keypoints
        else:
            raise ValueError(f"Incorrect primitive{self.primitive}")

        return world_points

    def _get_cv_image(self, idx):
        image = cv2.imread(self.color_images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.scene)

    def _sort_keypoints(self, keypoints_I):
        """
        Sorts the rectangle keypoints such that the are ordered top left, top right,
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
        bottom right, bottom left, as rectangles can be oriented arbitrarily.
        Might have to make this configurable and visible in Studio, as this might vary by application.
        Bounding boxes are assumed to have a specific orientation.
        """

        for rectangle in self.scene.rectangles:
            index = 4 * rectangle.class_id
            keypoints = keypoints_I[index:index+4]
            y_order = np.argsort(keypoints[:, 1])
            top_points = np.stack([keypoints[y_order[0]], keypoints[y_order[1]]])
            bottom_points = np.stack([keypoints[y_order[2]], keypoints[y_order[3]]])
            x_order_top = np.argsort(top_points[:, 0])
            x_order_bottom = np.argsort(bottom_points[:, 0])

            top_left = top_points[x_order_top[0]]
            top_right = top_points[x_order_top[1]]
            bottom_right = bottom_points[x_order_bottom[1]]
            bottom_left = bottom_points[x_order_bottom[0]]
            keypoints_I[index:index+4] = np.stack([
                top_left,
                top_right,
                bottom_right,
                bottom_left])
        return keypoints_I

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        heatmaps = []

        T_CW = np.linalg.inv(self.scene.poses[idx])
        keypoint_positions_C = transform(T_CW, self.keypoints_W)
        keypoint_positions_I = self.camera.project(keypoint_positions_C)
        if self.primitive == "rectangle":
            keypoint_positions_I = self._sort_keypoints(keypoint_positions_I)

        cv_image = self._get_cv_image(idx)
        if not self.eval:
            transformed = self.transform(image=cv_image, keypoints=keypoint_positions_I)
            cv_image = transformed["image"]
            keypoint_positions_I = transformed["keypoints"]
        else:
            cv_image = cv2.resize(cv_image, (self.image_width, self.image_height))

        cv_image = cv_image.astype(np.float32)
        np_image = np.transpose(cv_image/255.0, [2, 0, 1])

        for point_2d_I, point_3d in zip(keypoint_positions_I, self.keypoints_W):
            width_scale = self.out_width / self.image_width
            height_scale = self.out_height / self.image_height
            point_2d_I = [point_2d_I[0]*width_scale, point_2d_I[1]*height_scale]

            #NOTE: eval scenes that are not in real-world scale may need to use some constant for the distance
            point_3d_C = transform(T_CW, point_3d[None])[0]
            #TODO: possibly larger value
            diagonal_fraction = 0.3
            top_point = self.map_camera.project((point_3d_C  - I[1] * diagonal_fraction)[None])[0]
            bottom_point = self.map_camera.project((point_3d_C  + I[1] * diagonal_fraction)[None])[0]
            size = np.linalg.norm(top_point - bottom_point)
            lengthscale = np.sqrt(size**2/20.0)

            if self.eval:
                lengthscale = 5
            heatmap = get_heatmap(point_2d_I, self.out_width, self.out_height, lengthscale)
            heatmaps.append(heatmap)

        heatmaps = np.array(heatmaps)
        return torch.from_numpy(np_image).float(), torch.from_numpy(heatmaps).float()

class StrayKeypointDetectionDataset(ConcatDataset):
    def __init__(self, scene_paths, *args, **kwargs):
        scenes = []
        for scene_path in scene_paths:
            scenes.append(StrayKeypointScene(scene_path, *args, **kwargs))
        super().__init__(scenes)


class KeypointSceneData(pl.LightningDataModule):
    def __init__(self, train_dirs, eval_dirs, train_batch_size, eval_batch_size,
            num_workers, config, image_size, out_size):
        super().__init__()
        self.train_dirs = train_dirs
        self.eval_dirs = eval_dirs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.out_size = out_size
        self.config = config

    def train_dataloader(self):
        dataset = StrayKeypointDetectionDataset(self.train_dirs, self.config, self.image_size, self.out_size)

        return torch.utils.data.DataLoader(dataset,
                    num_workers=self.num_workers,
                    batch_size=self.train_batch_size,
                    persistent_workers=True if self.num_workers > 0 else False,
                    pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        dataset = StrayKeypointDetectionDataset(self.eval_dirs, self.config, self.image_size, self.out_size, eval=True)

        return torch.utils.data.DataLoader(dataset,
                    num_workers=self.num_workers,
                    batch_size=self.eval_batch_size,
                    persistent_workers=True if self.num_workers > 0 else False,
                    pin_memory=torch.cuda.is_available())
