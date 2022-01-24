import os
import cv2
import numpy as np
from stray.util.export import bbox_2d_from_bbox_3d, bbox_2d_from_mask, validate_segmentations
from stray.util.scene import Scene
import click
import pycocotools.mask as mask_util
import pickle

INSTANCE_COLORS = [
                (0.85098039, 0.37254902, 0.00784314),
                (0.10588235, 0.61960784, 0.46666667),
                (0.49803922, 0.78823529, 0.49803922),
                (0.74509804, 0.68235294, 0.83137255),
                (0.99215686, 0.75294118, 0.5254902),
                (1., 1., 0.6),
                (0.21960784, 0.42352941, 0.69019608),
                (0.94117647, 0.00784314, 0.49803922),
                (0.74901961, 0.35686275, 0.09019608),
                (0.90588235, 0.16078431, 0.54117647)
]

def render_keypoints(image, scene, i):
    T_WC = scene.poses[i]
    T_CW = np.linalg.inv(T_WC)
    world_keypoints = [keypoint.position for keypoint in scene.keypoints]
    image_keypoints = scene.camera().project(np.array(world_keypoints), T_CW)

    for keypoint_2d, scene_kp in zip(image_keypoints, scene.keypoints):
        instance_id = scene_kp.instance_id
        instance_color = INSTANCE_COLORS[instance_id]
        color = (int(255*instance_color[2]), int(255*instance_color[1]), int(255*instance_color[0]))
        image = cv2.circle(image, (int(keypoint_2d[0]), int(keypoint_2d[1])), 10, color, -1)
        image = cv2.putText(image, str(instance_id), (int(keypoint_2d[0]), int(keypoint_2d[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return image

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
@click.argument('scenes', nargs=-1)
@click.option('--bbox', default=False, is_flag=True, help='Show 2D bounding boxes')
@click.option('--keypoints', default=False, is_flag=True, help='Show keypoints')
@click.option('--segmentation', default=False, is_flag=True, help='Show segmentations')
@click.option('--bbox-from-mask', default=False, is_flag=True, help='Use the segmentatin mask to determine the 2D bounding box')
@click.option('--save', default=False, is_flag=True, help='Save labeled examples to <scene>/labeled_examples.')
@click.option('--rate', '-r', default=30.0, help="Frames per second to show frames.")
@click.option('--scale', '-s', default=1.0, help="Scale images.")
def main(**flags):
    title = "Stray Label Show"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    print("Playing through images.", end="\r")
    paused = False
    stop = False
    wait_time = int(1000.0 / flags["rate"])
    scene_paths = [scene_path for scene_path in flags["scenes"] if os.path.isdir(scene_path)]
    for scene_path in scene_paths:
        try:
            skip = False
            if stop:
                break
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
                if skip:
                    break
                filename = os.path.basename(image_path)
                image_idx = int(filename.split(".jpg")[0])

                print(f"Image {filename}" + " " * 10, end='\r')
                image = cv2.imread(image_path)

                if flags["bbox"]:
                    image = render_bboxes(flags, image, scene, image_idx)
                if flags["segmentation"]:
                    image = render_segmentations(scene, image, image_idx, colors)
                if flags["keypoints"]:
                    image = render_keypoints(image, scene, image_idx)

                image = cv2.resize(image, (int(image.shape[1]*flags["scale"]), int(image.shape[0]*flags["scale"])))

                if flags["save"]:
                    cv2.imwrite(os.path.join(labeled_save_path, os.path.basename(image_path.rstrip("/"))), image)
    
                cv2.imshow(title, image)
                key = cv2.waitKey(wait_time)

                if key == ord('q'):
                    stop = True
                    break
                if key == ord('s'):
                    skip = True
                    break

                elif key == ord(' '):
                    paused = not paused
                

                while paused:
                    key = cv2.waitKey(wait_time)
                    if key == ord(' '):
                        paused = not paused
                    if key == ord('q'):
                        stop = True
                        break
                    if key == ord('s'):
                        skip = True
                        break
        except Exception as e:
            print(f"Error: {e}")
            print(f"Skipping scene {scene_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

