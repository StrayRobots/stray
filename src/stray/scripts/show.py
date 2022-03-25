import os
import cv2
import numpy as np
from stray.export import bbox_2d_from_bbox_3d, bbox_2d_from_mask, validate_segmentations
from stray.scene import Scene
import click
import pycocotools.mask as mask_util
import pickle

def render_segmentations(scene, image, i, colors):
    for instance_id, _ in enumerate(scene.bounding_boxes):
        with open(os.path.join(scene.scene_path, "segmentation", f"instance_{instance_id}", f"{i:06}.pickle"), 'rb') as handle:
            segmentation = pickle.load(handle)
            mask = mask_util.decode(segmentation)
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)*255
        color = colors[instance_id]
        image[(rgb_mask==255).all(-1)] = color
    return image

def render_bboxes(flags, image, scene, i):
    for j, bbox in enumerate(scene.bounding_boxes):
        if flags["bbox_from_mask"]:
            bbox_flat = bbox_2d_from_mask(scene, j, i)
        else:
            bbox_flat = bbox_2d_from_bbox_3d(scene.camera(), scene.poses[i], bbox)
        bbox_np = np.array(bbox_flat).round().astype(np.int).reshape(2, 2)
        upper_left = bbox_np[0]
        lower_right = bbox_np[1]
        cv2.rectangle(image, tuple(upper_left - 3), tuple(lower_right + 3), (130, 130, 235), 2)
    return image

@click.command()
@click.argument('dataset', nargs=-1)
@click.option('--bbox', default=False, is_flag=True, help='Show 2D bounding boxes')
@click.option('--segmentation', default=False, is_flag=True, help='Show segmentations')
@click.option('--bbox-from-mask', default=False, is_flag=True, help='Use the segmentatin mask to determine the 2D bounding box')
@click.option('--save', default=False, is_flag=True, help='Save labeled examples to <scene>/labeled_examples.')
@click.option('--rate', '-r', default=30.0, help="Frames per second to show frames.")
def main(**flags):
    title = "Stray Label Show"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    print("Playing through images.", end="\r")
    paused = False
    wait_time = int(1000.0 / flags["rate"])
    scene_paths = [scene_path for scene_path in flags["dataset"] if os.path.isdir(scene_path)]

    for scene_path in scene_paths:
        if not os.path.isdir(scene_path):
            continue
        if flags["save"]:
            labeled_save_path = os.path.join(scene_path, "labeled_examples")
            os.makedirs(labeled_save_path, exist_ok=True)

        scene = Scene(scene_path)

        if flags["segmentation"] or flags["bbox_from_mask"]:
            validate_segmentations(scene)
        
        colors = [np.random.randint(0, 255, size=3) for _ in scene.bounding_boxes]
        for image_path in scene.get_image_filepaths():
            filename = os.path.basename(image_path)
            image_idx = int(filename.split(".jpg")[0])

            print(f"Image {filename}" + " " * 10, end='\r')
            image = cv2.imread(image_path)

            if flags["bbox"]:
                image = render_bboxes(flags, image, scene, image_idx)
            if flags["segmentation"]:
                image = render_segmentations(scene, image, image_idx, colors)
            if flags["save"]:
                cv2.imwrite(os.path.join(labeled_save_path, os.path.basename(image_path.rstrip("/"))), image)
 
            cv2.imshow(title, image)
            key = cv2.waitKey(wait_time)
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused

            while paused:
                key = cv2.waitKey(wait_time)
                if key == ord(' '):
                    paused = not paused
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

