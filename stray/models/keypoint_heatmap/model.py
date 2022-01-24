import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenetv2 import ConvBNActivation

class KeypointHeatmapModel(torch.nn.Module):
    def __init__(self, num_heatmaps=4, dropout=0.2):
        super(KeypointHeatmapModel, self).__init__()
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
            nn.Conv2d(128, num_heatmaps, kernel_size=1, stride=1, bias=True),
        )

        self.heatmap_head[-1].bias.data = torch.log(torch.ones(num_heatmaps) * 0.01)


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

