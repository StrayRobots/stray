import numpy as np
import os
import pickle
import pycocotools.mask as mask_util
import cv2

MISSING_SEGMENTATIONS = 1
INCORRECT_NUMBER_OF_SEGMENTATIONS = 2

def validate_segmentations(scene):
    length = len(scene.get_image_filepaths())
    for bbox_id in range(len(scene.bounding_boxes)):
        segmentation_path = os.path.join(scene.scene_path, "segmentation", f"instance_{bbox_id}")
        if not os.path.exists(segmentation_path):
            print(f"Missing segmentations at {segmentation_path}")
            exit(MISSING_SEGMENTATIONS)
        elif len([f for f in os.listdir(segmentation_path) if ".pickle" in f]) != length:
            print(f"Wrong number of segmentations at {segmentation_path}")
            exit(INCORRECT_NUMBER_OF_SEGMENTATIONS)


def bbox_2d_from_mesh(camera, T_WC, object_mesh):
    T_CW = np.linalg.inv(T_WC)
    vertices = object_mesh.vertices
    image_points = camera.project(vertices, T_CW)
    upper_left = image_points.min(axis=0)
    lower_right = image_points.max(axis=0)
    return upper_left.tolist() + lower_right.tolist()

def get_bbox_3d_corners(camera, T_WC, bbox_3d):
    T_CW = np.linalg.inv(T_WC)
    size = bbox_3d.dimensions
    corners_world = []
    for x_bbox in [-size[0]/2, size[0]/2]:
        for y_bbox in [-size[1]/2, size[1]/2]:
            for z_bbox in [-size[2]/2, size[2]/2]:
                corners_world.append(bbox_3d.position + bbox_3d.orientation.as_matrix()@np.array([x_bbox, y_bbox, z_bbox]))
    image_points = camera.project(np.array(corners_world), T_CW)
    return image_points

def bbox_2d_from_bbox_3d(camera, T_WC, bbox_3d):
    image_points = get_bbox_3d_corners(camera, T_WC, bbox_3d)
    upper_left = image_points.min(axis=0)
    lower_right = image_points.max(axis=0)
    return upper_left.tolist() + lower_right.tolist()

def bbox_2d_from_mask(scene, instance_id, i):
    with open(os.path.join(scene.scene_path, "segmentation", f"instance_{instance_id}", f"{i:06}.pickle"), 'rb') as handle:
        segmentation = pickle.load(handle)
    mask = mask_util.decode(segmentation)
    x,y,w,h = cv2.boundingRect(mask)
    return [x,y,x+w,y+h]