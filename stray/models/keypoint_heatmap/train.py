from stray.models.keypoint_heatmap.model import KeypointHeatmapModel
import torch.optim as optim
import numpy as np
from stray.losses.keypoint_heatmap_loss import KeypointHeatmapLoss
import pytorch_lightning as pl
import torch
from stray.util.visualization_utils import get_single_keypoint_heatmap_image, get_keypoint_heatmap_image
import cv2
import torch

def log_images(logger, step, log_type, image, gt_heatmaps, p_heatmaps):
    cv_image = np.ascontiguousarray(np.moveaxis(image*255, 0, -1), dtype=np.uint8)
    height, width, _ = cv_image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (np_gt_heatmap, np_p_heatmap) in enumerate(zip(gt_heatmaps, p_heatmaps)):
        cv_p_heatmap = get_single_keypoint_heatmap_image(np_p_heatmap, width, height)
        cv_gt_heatmap = get_single_keypoint_heatmap_image(np_gt_heatmap, width, height)

        p_image = cv2.addWeighted(cv_image, 0.65, cv_p_heatmap, 0.35, 0)
        gt_image = cv2.addWeighted(cv_image, 0.65, cv_gt_heatmap, 0.35, 0)
        p_peak = np.unravel_index(np_p_heatmap.argmax(), (80, 60))
        gt_peak = np.unravel_index(np_gt_heatmap.argmax(), (80, 60))

        p_image = cv2.putText(p_image, str(i), (int(p_peak[1]*8), int(p_peak[0]*8)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        gt_image = cv2.putText(gt_image, str(i), (int(gt_peak[1]*8), int(gt_peak[0]*8)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        p_image = np.transpose(cv2.cvtColor(p_image, cv2.COLOR_RGB2BGR), [2, 0, 1])
        gt_image = np.transpose(cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR), [2, 0, 1])

        logger.experiment.add_image(f'{log_type}_p_corners/{i}', p_image, step)
        logger.experiment.add_image(f'{log_type}_gt_corners/{i}', gt_image, step)

    cv_p_heatmaps = get_keypoint_heatmap_image(p_heatmaps, width, height)
    cv_gt_heatmaps = get_keypoint_heatmap_image(gt_heatmaps, width, height)

    p_image = cv2.addWeighted(cv_image, 0.65, cv_p_heatmaps, 0.35, 0)
    gt_image = cv2.addWeighted(cv_image, 0.65, cv_gt_heatmaps, 0.35, 0)

    p_image = np.transpose(cv2.cvtColor(p_image, cv2.COLOR_RGB2BGR), [2, 0, 1])
    gt_image = np.transpose(cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR), [2, 0, 1])

    logger.experiment.add_image(f'{log_type}_p_corners/all', p_image, step)
    logger.experiment.add_image(f'{log_type}_gt_corners/all', gt_image, step)


class KeypointHetmapTrainModule(pl.LightningModule):
    def __init__(self, lr, num_heatmaps=4, end_epoch_callback=None):
        super().__init__()
        self.lr = lr
        self.model = KeypointHeatmapModel(num_heatmaps)
        self.loss_function = KeypointHeatmapLoss()
        self.save_hyperparameters()
        self.end_epoch_callback = end_epoch_callback

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, gt_heatmaps = batch
        p_heatmaps = self(images)
        loss = self.loss_function(p_heatmaps, gt_heatmaps)
        self.log('train_loss', loss)
        if batch_idx == 0:
            index = np.random.randint(0, p_heatmaps.shape[0])
            image = images[index].detach().cpu().numpy()
            p_heatmap = torch.sigmoid(p_heatmaps)[index]
            np_p_heatmaps = p_heatmap.detach().cpu().numpy()
            np_gt_heatmaps = gt_heatmaps[index].detach().cpu().numpy()
            log_images(self.logger, self.global_step, "train", image, np_gt_heatmaps, np_p_heatmaps)
        return loss

    def validation_step(self, batch, batch_idx):
        images, gt_heatmaps = batch
        p_heatmaps = self(images)
        loss = self.loss_function(p_heatmaps, gt_heatmaps)
        self.log('val_loss', loss)
        if batch_idx == 0:

            #Save images
            index = np.random.randint(0, p_heatmaps.shape[0])
            image = images[index].detach().cpu().numpy()
            p_heatmap = torch.sigmoid(p_heatmaps)[index]
            np_p_heatmaps = p_heatmap.detach().cpu().numpy()
            np_gt_heatmaps = gt_heatmaps[index].detach().cpu().numpy()
            log_images(self.logger, self.global_step, "val", image, np_gt_heatmaps, np_p_heatmaps)

        return loss

    def validation_epoch_end(self, _):
        if self.end_epoch_callback is not None:
            self.end_epoch_callback(self)


    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
