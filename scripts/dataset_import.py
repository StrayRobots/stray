import os
import click
import shutil
import numpy as np
import json
import csv
import cv2
from pathlib import Path
from skvideo import io

def _resize_and_rotate_camera_matrix(camera_matrix, scale_x, scale_y, rotate):
    if scale_x == 1.0 and scale_y == 1.0:
        return camera_matrix
    if rotate is not None:
        fy = camera_matrix[0, 0]
        fx = camera_matrix[1, 1]
        cy = camera_matrix[0, 2]
        cx = camera_matrix[1, 2]
    else:
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
    return np.array([[fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]])

def write_frames(dataset, every, rgb_out_dir, width, height, rotate):
    rgb_video = os.path.join(dataset, 'rgb.mp4')
    video = io.FFmpegReader(rgb_video)
    fps = video.inputfps
    for i, frame in enumerate(video.nextFrame()):
        if i % every != 0:
            continue
        print(f"Writing rgb frame {i:06}" + " " * 10, end='\r')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        full_width = frame.shape[1]
        full_height = frame.shape[0]
        frame = cv2.resize(frame, (width, height))
        frame_path = os.path.join(rgb_out_dir, f"{i:06}.jpg")
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        cv2.imwrite(frame_path, frame, params)
    video.close()
    print('\r')
    return full_width, full_height, fps

def resize_and_rotate_depth(frame, width, height, rotate):
    if rotate == "cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "ccw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    out = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    out[out < 10] = 0
    return out

def write_depth(dataset, every, depth_out_dir, width, height, rotate):
    depth_dir_in = os.path.join(dataset, 'depth')
    confidence_dir = os.path.join(dataset, 'confidence')
    files = sorted(os.listdir(depth_dir_in))
    for i, filename in enumerate(files):
        file_extension = filename.split('.')[-1]
        if file_extension not in ['npy', 'png'] or i % every != 0:
            continue
        number, _ = filename.split('.')
        depth_file = os.path.join(depth_dir_in, filename)
        confidence_file = os.path.join(confidence_dir, number + '.png')
        if file_extension == 'npy':
            depth = np.load(depth_file)
        else:
            depth = cv2.imread(depth_file, -1)
        if os.path.exists(confidence_file):
            confidence = cv2.imread(confidence_file)[:, :, 0]
        else:
            confidence = np.ones_like(depth_file, dtype=np.uint8) * 2
        depth[confidence < 2] = 0
        depth = resize_and_rotate_depth(depth, width, height, rotate)
        cv2.imwrite(os.path.join(depth_out_dir, number + '.png'), depth)

def write_intrinsics(dataset, out, width, height, full_width, full_height, rotate, fps):
    intrinsics = np.loadtxt(os.path.join(dataset, 'camera_matrix.csv'), delimiter=',')
    data = {}
    intrinsics_scaled = _resize_and_rotate_camera_matrix(intrinsics, width / full_width, height / full_height, rotate)
    data['intrinsic_matrix'] = [intrinsics_scaled[0, 0], 0.0, 0.0,
            0.0, intrinsics_scaled[1, 1], 0.0,
            intrinsics_scaled[0, 2], intrinsics_scaled[1, 2], 1.0]
    data['width'] = width
    data['height'] = height
    data['depth_scale'] = 1000.0
    data['fps'] = fps
    data['depth_format'] = 'Z16'
    with open(os.path.join(out, 'camera_intrinsics.json'), 'wt') as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def copy_rgb(scene_path, target_path):
    shutil.copy(os.path.join(scene_path, 'rgb.mp4'),
            os.path.join(target_path, 'rgb.mp4'))

def copy_imu(scene_path, target_path):
    imu_csv = os.path.join(scene_path, 'imu.csv')
    if os.path.exists(imu_csv):
        shutil.copy(imu_csv, os.path.join(target_path, 'imu.csv'))

def write_frame_data(scene_path, target_path, every=1):
    odometry_csv = os.path.join(scene_path, 'odometry.csv')
    with open(odometry_csv, 'rt') as in_file:
        reader = csv.reader(in_file)
        header = next(reader)
        with open(os.path.join(target_path, 'frames.csv'), 'wt') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(['timestamp', 'frame', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            for line in reader:
                timestamp = line[0]
                frame = line[1]
                if int(frame) % every != 0:
                    continue
                writer.writerow([timestamp, frame] + line[2:9])

def validate_scene_path(scene_path) -> str:
    """
    Checks if a path looks like a Scanner directory. Returns a fixed path, if for example the path
    refers to a subfile or directory in the scene folder. If scene_path is an actual scene path, then
    this is an identity function.

    throws ValueError if this doesn't look to be a scene folder.

    scene_path: str path to a potential path
    returns: str scene_path or fixed scene_path
    """
    def looks_like_scanner_scene(path):
        is_dir = path.is_dir()
        has_rgb = (path / "rgb.mp4").is_file()
        has_depth = (path / "depth").is_dir()
        has_intrinsics = (path / "camera_matrix.csv").is_file()
        return is_dir and has_rgb and has_depth and has_intrinsics

    path = Path(scene_path)
    if looks_like_scanner_scene(path):
        return scene_path
    elif looks_like_scanner_scene(path.parent):
        return str(path.parent)
    elif looks_like_scanner_scene(path.parent.parent):
        return str(path.parent.parent)
    raise ValueError(scene_path)

@click.command()
@click.argument('scenes', nargs=-1)
@click.option('--out', '-o', required=True, help="Dataset directory where to place the imported files.", type=str)
@click.option('--every', type=int, default=1, help="Keep only every n-th frame. 1 keeps every frame, 2 keeps every other and so forth.")
@click.option('--width', '-w', type=int, default=1920)
@click.option('--height', '-h', type=int, default=1440)
@click.option('--rotate', '-r', type=click.Choice(["cw", "ccw"]))
@click.option('--intrinsics', type=str, help="Path to the intrinsic parameters to use (for example calibrated parameters from stray calibration run). Defaults to factory parameters.")
def main(scenes, out, every, width, height, rotate, intrinsics):
    """
    Command for importing scenes from the Stray Scanner format to the Stray Dataset format.

    Usage: import <scanner-scenes> --out <output-dataset-folder>

    Each scene will be imported and converted into the dataset folder.
    """
    try:
        scenes = [validate_scene_path(p) for p in scenes]
    except ValueError as error:
        print(f"Path {error.args[0]} does not look like a Stray Scanner scene folder.")
        exit(1)

    os.makedirs(out, exist_ok=True)
    existing_scenes = os.listdir(out)
    for scene_path in scenes:
        if scene_path[-1] == os.path.sep:
            scene_path = scene_path[:-1]
        scene_base_name = os.path.basename(scene_path)
        if scene_base_name[0] == ".":
            continue
        if scene_base_name in existing_scenes:
            print(f"Scene {scene_base_name} exists already, skipping.")
            continue
        target_path = os.path.join(out, scene_base_name)
        print(f"Importing scene {scene_path} into {target_path}")

        rgb_out = os.path.join(target_path, 'color/')
        depth_out = os.path.join(target_path, 'depth/')
        os.makedirs(rgb_out)
        os.makedirs(depth_out)

        write_depth(scene_path, every, depth_out, width, height, rotate)
        full_width, full_height, fps = write_frames(
            scene_path, every, rgb_out, width, height, rotate)

        copy_rgb(scene_path, target_path)
        copy_imu(scene_path, target_path)
        write_frame_data(scene_path, target_path, every)

        if intrinsics is None:
            if os.path.exists(os.path.join(scene_path, 'camera_matrix.csv')):
                print("Writing factory intrinsics.", end='\n')
                write_intrinsics(scene_path, target_path, width,
                         height, full_width, full_height, rotate, fps)
            else:
                print("Warning: no camera matrix found, skipping.")
        else:
            print("Writing intrinsics.", end='\n')
            shutil.copy(intrinsics, os.path.join(target_path, 'camera_intrinsics.json'))

    print("Done.")

if __name__ == "__main__":
    main()


