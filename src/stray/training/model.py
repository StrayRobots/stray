import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
import torch
import numpy as np
from stray.training.loss import KeypointLoss
from torchvision import models
from torchvision.models.mobilenetv2 import ConvBNActivation
from stray.training import log_images

class KeypointNet(torch.nn.Module):
    def __init__(self, output_maps, dropout=0.2):
        super(KeypointNet, self).__init__()
        regnet = models.regnet_y_800mf(pretrained=True)
        self.backbone = nn.Sequential(regnet.stem, regnet.trunk_output)

        self.upsample = nn.Sequential(
            ConvBNActivation(784, 512, kernel_size=1, stride=1, padding=0),
            nn.Upsample(mode='bilinear', scale_factor=2.0, align_corners=False),
            ConvBNActivation(512, 256, kernel_size=3, stride=1, padding=1),
            nn.Upsample(mode='bilinear', scale_factor=2.0, align_corners=False),
        )
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        self.heatmap_head = nn.Sequential(
            ConvBNActivation(256, 128, kernel_size=3, stride=1),
            nn.Conv2d(128, output_maps, kernel_size=1, stride=1, bias=True),
        )

        self.heatmap_head[-1].bias.data = torch.log(torch.ones(4) * 0.01)


    def forward(self, x):
        features = self.backbone(x)
        features = self.upsample(features)
        features = self.heatmap_head(features)
        return features


    def eval(self, train=False):
        for net in self._get_networks():
           net.eval()

    def train(self, train=True):
        for net in self._get_networks():
           net.train()

    #Torch Script saving does not allow this to be a @property
    def _get_networks(self):
        return [
            self.backbone,
            self.upsample,
            self.dropout,
            self.heatmap_head
        ]


class KeypointTrainModule(pl.LightningModule):
    def __init__(self, num_keypoints, lr):
        super().__init__()
        self.lr = lr
        self.model = KeypointNet(num_keypoints)
        self.loss_function = KeypointLoss()
        self.save_hyperparameters()

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
            index = np.random.randint(0, p_heatmaps.shape[0])
            image = images[index].detach().cpu().numpy()
            p_heatmap = torch.sigmoid(p_heatmaps)[index]
            np_p_heatmaps = p_heatmap.detach().cpu().numpy()
            np_gt_heatmaps = gt_heatmaps[index].detach().cpu().numpy()
            log_images(self.logger, self.global_step, "val", image, np_gt_heatmaps, np_p_heatmaps)

        return loss


    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)