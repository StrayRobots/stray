import numpy as np
import cv2


def get_keypoint_heatmap_image(heatmaps, width, height):
    summed_heatmaps = np.clip(np.sum(heatmaps, axis=0), 0, 1)
    cv_heatmap = (summed_heatmaps*255).astype(np.uint8)
    cv_heatmap = cv2.applyColorMap(cv_heatmap, cv2.COLORMAP_JET)
    cv_heatmap = cv2.resize(cv_heatmap, (width, height))
    return cv_heatmap

def get_single_keypoint_heatmap_image(heatmap, width, height):
    cv_heatmap = (heatmap*255).astype(np.uint8)
    cv_heatmap = cv2.applyColorMap(cv_heatmap, cv2.COLORMAP_JET)
    cv_heatmap = cv2.resize(cv_heatmap, (width, height))
    return cv_heatmap