from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import math
import numpy as np
import cv2
from skspatial.objects import Plane, Line
import torch

def load_config(model_path, minimum_confidence):
    config = get_cfg()
    if os.path.isfile(os.path.join(model_path, "config.yaml")):
        config.merge_from_file(os.path.join(model_path, "config.yaml"))
    
    config.MODEL.WEIGHTS = os.path.join(model_path, "output", "model_final.pth")
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = minimum_confidence
    return config


class OrientedBoundingBoxDetector(object):
    def __init__(self, model, image_width, image_height, fx, fy, depth_scale, minimum_confidence=0.9, depth_box_scale=0.5):
        self.cfg = load_config(model, minimum_confidence)
        if (torch.cuda.is_available()):
            self.cfg.MODEL.DEVICE = "cuda:0"
        else:
            self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        self.image_width = image_width
        self.image_height = image_height
        self.fx = fx
        self.fy = fy
        self.depth_scale = depth_scale
        self.depth_box_scale = depth_box_scale

    def get_ray(self, x, y):
        aspect_ratio = self.image_width / self.image_height
        fovx = 2*math.atan(self.image_width/(2*self.fx))
        fovy= 2*math.atan(self.image_height/(2*self.fy))
        pX = (2.0 * ((x + 0.5) / self.image_width) - 1.0) * math.tan(fovx / 2.0) * aspect_ratio
        pY = (1.0 - 2.0 * ((y + 0.5) / self.image_height)) * math.tan(fovy / 2.0)
        vector = np.array([pX, -pY, 1])
        return vector / np.linalg.norm(vector)

    def get_point_3d_from_depth(self, instance, depth):
        x, y = instance["bounding_box"]["center"]
        width, height, rotation  = instance["bounding_box"]["width"], instance["bounding_box"]["height"], instance["bounding_box"]["rotation"]
        mask = np.float32(np.zeros((self.image_height, self.image_width, 3)))
        rect = ((x, y), (width, height), rotation)
        cv_box = cv2.boxPoints(rect)
        cv_box = np.int0(cv_box)
        cv2.fillConvexPoly(mask, cv_box, (0,255,0))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 100, 1, cv2.THRESH_BINARY)[1]
        mask = mask.astype(bool)
        masked_depth = depth[mask]
        length = np.mean(masked_depth[np.nonzero(masked_depth)])*self.depth_scale
        return self.get_ray(x, y)*length

    def get_point_3d_from_height(self, instance, z):
        x, y = instance["bounding_box"]["center"]
        plane_C = Plane(point=[0, 0, z], normal=[0, 0, -1])
        ray = self.get_ray(x, y)
        line_C = Line([0, 0, 0], ray)
        point_3d_C = plane_C.intersect_line(line_C)
        return np.array(point_3d_C)

    def __call__(self, color, depth, z=None):
        predictions = []
        raw_predictions = self.predictor(color)
        raw_bounding_boxes = raw_predictions["instances"].to("cpu").pred_boxes.tensor.numpy()
        categories = raw_predictions["instances"].to("cpu").pred_classes.numpy()
        confidences = raw_predictions["instances"].to("cpu").scores.numpy()

        for i, (bbox, category, confidence) in enumerate(zip(raw_bounding_boxes, categories, confidences)):
            instance = {
                "instance_id": i,
                "category_id": category,
                "confidence": confidence,
                "bounding_box": {
                    "center": [bbox[0], bbox[1]],
                    "width" : bbox[2],
                    "height": bbox[3],
                    "rotation": -bbox[4]
                }
            }
            if z is not None:
                point_3d = self.get_point_3d_from_height(instance, z)
            else:
                point_3d = self.get_point_3d_from_depth(instance, depth)

            instance["center_3d"] = point_3d
            predictions.append(instance)

        return predictions
