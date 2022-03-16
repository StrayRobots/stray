import numpy as np
import cv2
import torch.nn.functional as F

def spatial_softmax(predictions):
    """
    Convert predictions to heatmaps by normalizing over the 2d image.
    """
    N, C, H, W = predictions.shape
    return F.softmax(predictions.reshape(N, C, -1), dim=2).reshape(N, C, H, W)

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


I = np.eye(3)

def get_heatmap(center, width, height, lengthscale):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    y = y[:,np.newaxis]
    x_c = center[0]
    y_c = center[1]
    return np.exp(-4*np.log(2) * ((x-x_c)**2 + (y-y_c)**2) / lengthscale**2)

def transform(T, vectors):
    return (T[:3, :3] @ vectors[:, :, None])[:, :, 0] + T[:3, 3]





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