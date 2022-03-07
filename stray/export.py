import numpy as np
from stray.scene import Scene
import os
import pickle
import pycocotools.mask as mask_util
import cv2
from stray import linalg

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

def bbox_2d_from_pointcloud(camera, T_WC, pointcloud):
    # filter out points behind camera.
    T_CW = np.linalg.inv(T_WC)
    pc_local = linalg.transform_points(T_CW, pointcloud)
    pc = pointcloud[pc_local[:, 2] > 0.0]
    if pc.shape[0] == 0:
        return [0, 0, 0, 0]
    points2d = camera.project(pc, np.linalg.inv(T_WC))
    upper_left = points2d.min(axis=0)
    image_size = np.array(camera.size)
    upper_left = np.maximum(points2d.min(axis=0), 0)
    upper_left = np.minimum(upper_left, image_size)
    lower_right = np.maximum(points2d.max(axis=0), 0)
    lower_right = np.minimum(lower_right, image_size)
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


def compute_instance_keypoints(camera, T_WC, instance, instance_keypoints, max_num_keypoints=0):
    T_CW = np.linalg.inv(T_WC)
    world_keypoints = [instance.position + instance.orientation.as_matrix()@np.multiply(np.array(kp),instance.dimensions/2) for kp in instance_keypoints]
    image_keypoints = camera.project(np.array(world_keypoints), T_CW)
    flat_keypoints = []
    for image_point in image_keypoints:
        flat_keypoints.append(image_point[0])
        flat_keypoints.append(image_point[1])
        flat_keypoints.append(2)
    while len(flat_keypoints) < max_num_keypoints:
        flat_keypoints += [0, 0, 0]

    return flat_keypoints


def get_scene_dataset_metadata(scene_paths):
    instance_categories = []
    max_num_keypoints = 0
    for scene_path in scene_paths:
        scene = Scene(scene_path)
        metadata = scene.metadata
        for bbox_id in scene.bbox_categories:
            if not bbox_id in instance_categories:
                instance_categories.append(bbox_id)
            if metadata:
                bbox_metadata_instance = [instance for instance in metadata["instances"] if instance["instance_id"] == bbox_id]
                if len(bbox_metadata_instance) > 0:
                    num_keypoints = len(bbox_metadata_instance[0].get("keypoints", []))
                    max_num_keypoints = max(max_num_keypoints, num_keypoints)

    instance_category_mapping = {}
    for i, instance_id in enumerate(instance_categories):
        instance_category_mapping[f"instance_{instance_id}"] = i

    return {
        'max_num_keypoints': max_num_keypoints,
        'instance_category_mapping': instance_category_mapping,
        'thing_classes': list(instance_category_mapping.keys())
    }

def get_detectron2_dataset_function(scene_paths, dataset_metadata, use_bbox_from_mask, use_segmentation):
    def inner():
        examples = []
        for scene_path in scene_paths:
            scene = Scene(scene_path)
            if use_segmentation:
                validate_segmentations(scene)
            width, height = scene.image_size()
            images = scene.get_image_filepaths()
            metadata = scene.metadata
            image_id = 0
            max_num_keypoints = dataset_metadata["max_num_keypoints"]
            for image_path in images:
                filename = os.path.basename(image_path)
                image_idx = int(filename.split(".jpg")[0])

                annotations = []
                for bbox_id, bbox in enumerate(scene.bounding_boxes):
                    if use_bbox_from_mask:
                        bbox_flat = bbox_2d_from_mask(scene, bbox_id, image_idx)
                    else:
                        bbox_flat = bbox_2d_from_bbox_3d(scene.camera(), scene.poses[image_idx], bbox)
                    annotation = {
                        'category_id': dataset_metadata['instance_category_mapping'][f"instance_{bbox.instance_id}"],
                        'bbox': bbox_flat,
                        'bbox_mode': 0
                    }
                    if use_segmentation:
                        with open(os.path.join(scene_path, "segmentation", f"instance_{bbox_id}", f"{image_idx:06}.pickle"), 'rb') as handle:
                            segmentation = pickle.load(handle)
                            annotation["segmentation"] = segmentation

                    if metadata:
                        bbox_metadata_instance = [instance for instance in metadata["instances"] if instance["instance_id"] == bbox.instance_id]
                        bbox_keypoints = bbox_metadata_instance[0].get("keypoints", [])
                        if len(bbox_metadata_instance) > 0 and len(bbox_keypoints) > 0:
                            image_keypoints = compute_instance_keypoints(scene.camera(), scene.poses[image_idx], bbox, bbox_metadata_instance[0]["keypoints"], max_num_keypoints)
                            annotation["keypoints"] = image_keypoints
                        else:
                            annotation["keypoints"] = [0 for _ in range(max_num_keypoints*3)]
                    annotations.append(annotation)

                examples.append({
                    'file_name': image_path,
                    'image_id': image_id,
                    'height': height,
                    'width': width,
                    'annotations': annotations
                })
                image_id += 1
        return examples
    return inner

