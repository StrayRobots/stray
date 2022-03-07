import os
from stray.scene import Scene
from stray.renderer import Renderer
import numpy as np
import pycocotools.mask as mask_util
import pickle


def write_segmentation_masks(scene_path):
    scene = Scene(scene_path)
    renderer = Renderer(scene)
    segmentation_parent_path = os.path.join(scene_path, "segmentation")
    os.makedirs(segmentation_parent_path, exist_ok=True)
    for bbox_id, bbox in enumerate(scene.bounding_boxes):
        segmentation_path = os.path.join(segmentation_parent_path, f"instance_{bbox_id}")
        os.makedirs(segmentation_path, exist_ok=True)
        renderer.add_scene_instance(bbox)
        for i in range(0, len(scene), 1):
            print(f"Processing frame {i:06}", end='\r')
            mask = renderer.render_segmentation(i)
            segmentation = mask_util.encode(np.asarray(mask, order="F"))
            with open(os.path.join(segmentation_path, f"{i:06}.pickle"), 'wb') as handle:
                pickle.dump(segmentation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved segmetations to {segmentation_path} for instance {bbox_id}")
        renderer.clear_scene_instances()

