import torch
import numpy as np
from sklearn.cluster import MeanShift

use_cuda = torch.cuda.is_available()

class KeypointHeatmapDetector:
    def __init__(self, model):
        image_size = np.array([640., 480.])
        output_size = np.array([80., 60.])
        self.output_coordinate_scaling_factor = image_size / output_size
        self.model = model
        if use_cuda:
            self.model = self.model.cuda()
        self.bandwidth = 0.05 * output_size[0]

    def __call__(self, x):
        single = False
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if len(x.shape) == 3:
            # Single image, add batch dimension.
            x = x[None]
            single = True

        if x.shape[1] != 3:
            # Color is not in third dimension.
            x = x.transpose(1, 3).transpose(2, 3)

        x = x.to(torch.float32) / 255.0
        with torch.inference_mode():
            if use_cuda:
                x = x.cuda()
            heatmaps = self.model(x)
            heatmaps = torch.sigmoid(heatmaps)
        extracted = self._extract_keypoints(heatmaps.cpu().numpy())
        if single:
            return extracted[0]
        else:
            return extracted

    def _extract_keypoints(self, batch):
        out = []
        for heatmaps in batch:
            mean_shift = MeanShift(bandwidth=self.bandwidth)
            keypoints = []
            for i, heatmap in enumerate(heatmaps):
                points = np.argwhere(heatmap > 0.1)[:, [1, 0]]
                if len(points) == 0:
                    keypoints.append([])
                    continue
                mean_shift.fit(points)
                kp = mean_shift.cluster_centers_
                kp = kp * self.output_coordinate_scaling_factor
                keypoints.append(kp)

            out.append({
                'keypoints': keypoints,
                'heatmaps': heatmaps
            })

        return out




