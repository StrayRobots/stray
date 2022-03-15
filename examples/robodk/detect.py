import multiprocessing
import numpy as np
import utils
import queue
from PIL import Image
from constants import *
from stray.camera import Camera

DEPTH_DETECT_RADIUS = 50.0

def load_depth(path):
    """
    Loads depth images from disk captured by RoboDK.
    """
    depth_image = np.fromfile(path, dtype='>u4')
    w, h = depth_image[:2]
    depth_image = np.flipud(np.reshape(depth_image[2:], (h, w)))
    depth_image = np.floor(depth_image * 0.35).astype(np.float32)
    return depth_image / 1000.0

def compute_depth_estimate(depth_frame, detection_center):
    """
    Given a depth frame and the detection point, estimates the depth at that point.
    """
    height, width = depth_frame.shape
    # Get a square centered at the detected centered at the center of the detections
    # to estimate how far away it is.
    x_start = max(round(detection_center[0] - DEPTH_DETECT_RADIUS * 0.5), 0)
    x_end = min(round(detection_center[0] + DEPTH_DETECT_RADIUS * 0.5), width)
    y_start = max(round(detection_center[1] - DEPTH_DETECT_RADIUS * 0.5), 0)
    y_end = min(round(detection_center[1] + DEPTH_DETECT_RADIUS * 0.5), height)
    depth_readings = depth_frame[y_start:y_end, x_start:x_end]
    return depth_readings[depth_readings > 0.0].min()

def detection_loop(queue_in, queue_out, model):
    """
    The main object detection loop.
    Reads images from the queue, runs object detection
    and pushes pick points to the output queue.
    """
    import torch
    from stray.detection import KeypointDetector
    model = KeypointDetector(model, lengthscale=50.0)
    camera = Camera((IMAGE_WIDTH, IMAGE_HEIGHT),
            utils.compute_camera_matrix(FIELD_OF_VIEW, IMAGE_WIDTH, IMAGE_HEIGHT))

    while True:
        timestamp, rgb_image_path, depth_image_path = queue_in.get()
        depth_frame = load_depth(depth_image_path)

        image = np.array(Image.open(rgb_image_path))
        out = model(image)
        keypoints = out['keypoints']
        # Check that all four corners were detected.
        if not sum([len(points) > 0 for points in keypoints]) == 4:
            continue

        keypoints = np.stack([points[0] for points in keypoints])
        center = keypoints.mean(axis=0)
        z_pick_point = compute_depth_estimate(depth_frame, center)
        point = camera.unproject(center[None], z_pick_point[None])[0]

        center_point = camera.project(point[None])

        # Replace old messages with latest.
        while queue_out.full():
            queue_out.get()

        queue_out.put((timestamp, point))


class Detector:
    """
    A helper class to run the detection loop in a background process.
    """
    def __init__(self, model):
        self.queue_in = multiprocessing.Queue(3)
        self.queue_out = multiprocessing.Queue(1)
        self.process = multiprocessing.Process(target=detection_loop,
                args=(self.queue_in, self.queue_out, model), daemon=True)
        self.process.start()

    def full(self):
        return self.queue_in.full()

    def push(self, item):
        self.queue_in.put(item)

    def get(self):
        try:
            return self.queue_out.get(False)
        except queue.Empty:
            return None

    def clear(self):
        while not self.queue_out.empty():
            self.queue_out.get()
        while not self.queue_in.empty():
            self.queue_in.get()




