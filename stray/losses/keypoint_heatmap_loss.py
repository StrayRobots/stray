import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class KeypointHeatmapLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, p_heatmaps, gt_heatmaps):
        activated = torch.sigmoid(p_heatmaps)
        loss =  F.mse_loss(activated, gt_heatmaps)
        return loss