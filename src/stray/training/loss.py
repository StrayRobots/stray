import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from stray.training import spatial_softmax

class IntegralKeypointLoss(_Loss):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.device = device
        self.indices = torch.zeros((2, size[1], size[0]), device=self.device, dtype=torch.float)
        for i in range(size[1]):
            for j in range(size[0]):
                self.indices[:, i, j] = torch.tensor((j, i), device=self.device, dtype=torch.float) + 0.5
        self.indices = self.indices[None]

    def _integrate_maps_unproject_points(self, heatmap, depthmap, K):
        heatmap_double = heatmap.repeat(1, 2, 1, 1)
        points2d = (self.indices * heatmap_double).sum(dim=[2, 3]) / heatmap.sum(dim=[2, 3])
        depth = depthmap * heatmap
        depth = depth.sum(dim=[1, 2, 3])

        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]

        x = (points2d[..., 0] - cx) / fx
        y = (points2d[..., 1] - cy) / fy
        xyz = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        return xyz * depth[:, None]

class IntegralKeypointLoss(_Loss):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.device = device
        self.indices = torch.zeros((2, size[1], size[0]), device=self.device, dtype=torch.float)
        for i in range(size[1]):
            for j in range(size[0]):
                self.indices[:, i, j] = torch.tensor((j, i), device=self.device, dtype=torch.float) + 0.5
        self.indices = self.indices[None]

    def _integrate_maps_unproject_points(self, heatmap, depthmap, K):
        heatmap_double = heatmap.repeat(1, 2, 1, 1)
        points2d = (self.indices * heatmap_double).sum(dim=[2, 3]) / heatmap.sum(dim=[2, 3])
        depth = depthmap * heatmap
        depth = depth.sum(dim=[1, 2, 3])

        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]

        x = (points2d[..., 0] - cx) / fx
        y = (points2d[..., 1] - cy) / fy
        xyz = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        return xyz * depth[:, None]

    def forward(self, heatmap, depthmap, K, gt_points):
        points3d = self._integrate_maps_unproject_points(heatmap, depthmap, K)
        return F.l1_loss(points3d, gt_points)

class BoundingBoxLoss(_Loss):
    def __init__(self, size, center_weight=1.0, heatmap_weight=1.0, corner_weight=1.0, device='cpu'):
        super().__init__()
        self.integral_loss = IntegralKeypointLoss(size, device=device)
        self.center_weight = center_weight
        self.heatmap_weight = heatmap_weight
        self.corner_weight = corner_weight

    def forward(self, p_heatmap, p_depth, p_corners, gt_heatmap, gt_corners, Ks, gt_points):
        softmaxed = spatial_softmax(p_heatmap)
        center_loss = self.integral_loss(softmaxed, p_depth, Ks, gt_points)
        l2_heatmap_loss = F.mse_loss(softmaxed, gt_heatmap)
        where_support = (gt_heatmap > 0.0).repeat(1, 16, 1, 1)
        corner_loss = F.l1_loss(p_corners[where_support], gt_corners[where_support])
        return self.center_weight * center_loss, self.heatmap_weight * l2_heatmap_loss, self.corner_weight * corner_loss

class KeypointLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, p_heatmaps, gt_heatmaps):
        activated = torch.sigmoid(p_heatmaps)
        loss =  F.mse_loss(activated, gt_heatmaps)
        return loss

        points3d = self._integrate_maps_unproject_points(heatmap, depthmap, K)
        return F.l1_loss(points3d, gt_points)

class BoundingBoxLoss(_Loss):
    def __init__(self, size, center_weight=1.0, heatmap_weight=1.0, corner_weight=1.0, device='cpu'):
        super().__init__()
        self.integral_loss = IntegralKeypointLoss(size, device=device)
        self.center_weight = center_weight
        self.heatmap_weight = heatmap_weight
        self.corner_weight = corner_weight

    def forward(self, p_heatmap, p_depth, p_corners, gt_heatmap, gt_corners, Ks, gt_points):
        softmaxed = spatial_softmax(p_heatmap)
        center_loss = self.integral_loss(softmaxed, p_depth, Ks, gt_points)
        l2_heatmap_loss = F.mse_loss(softmaxed, gt_heatmap)
        where_support = (gt_heatmap > 0.0).repeat(1, 16, 1, 1)
        corner_loss = F.l1_loss(p_corners[where_support], gt_corners[where_support])
        return self.center_weight * center_loss, self.heatmap_weight * l2_heatmap_loss, self.corner_weight * corner_loss

class KeypointLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, p_heatmaps, gt_heatmaps):
        activated = torch.sigmoid(p_heatmaps)
        loss =  F.mse_loss(activated, gt_heatmaps)
        return loss
