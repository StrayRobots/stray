import os
import click
import shutil
import uuid
import hashlib
from pathlib import Path

SCENE_DOES_NOT_EXIST = 1
NOT_SCENE_DIR = 2

def validate_scene(scene):
    if not os.path.exists(scene):
        print(f"{scene} does not exist.")
        exit(SCENE_DOES_NOT_EXIST)
    if not os.path.isdir(scene):
        print(f"{scene} is not a directory")
        exit(NOT_SCENE_DIR)
    if not os.path.exists(os.path.join(scene, 'color')) or not os.path.exists(os.path.join(scene, 'depth')):
        print("Does not look like a scene directory")
        exit(NOT_SCENE_DIR)

def new_hash():
    return hashlib.md5(uuid.uuid4().bytes).hexdigest()[:10]

def frame_number(filename):
    return int(filename.split('.')[0])

def copy_metadata(scene, new_dir):
    shutil.copy2(os.path.join(scene, 'camera_intrinsics.json'), os.path.join(new_dir, 'camera_intrinsics.json'))

def move_images(frames, directory):
    depth_dir = os.path.join(directory, 'depth')
    rgb_dir = os.path.join(directory, 'color')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    for rgb, depth in frames:
        number = os.path.basename(rgb).split('.')[0]
        print(f"Moving frame {number}", end='\r')
        if os.path.exists(rgb):
            shutil.move(rgb, os.path.join(rgb_dir, os.path.basename(rgb)))
        if os.path.exists(depth):
            shutil.move(depth, os.path.join(depth_dir, os.path.basename(depth)))
    print("")

@click.command()
@click.argument('scene', nargs=1)
@click.option('--start', '-s', type=int, default=0)
@click.option('--end', '-e', type=int, default=None)
def main(scene, start, end):
    """
    Command for cutting scenes. Frames between --start and --end are kept. Frames outside of this are moved into a new scene folder that is created.

    Usage: stray dataset cut <scanner-scene> [--start 0] --end 1000
    """
    validate_scene(scene)

    new_dir = os.path.join(str(Path(scene).parent), new_hash())
    rgb_dir = os.path.join(scene, 'color')
    depth_dir = os.path.join(scene, 'depth')
    rgb_frames = sorted([f for f in os.listdir(rgb_dir) if '.jpg' in f])
    depth_frames = sorted([f for f in os.listdir(depth_dir) if '.png' in f])

    if end is None:
        end = min(len(rgb_frames), len(depth_frames))

    new_frames = []
    for i in list(range(0, start)) + list(range(end, max(frame_number(rgb_frames[-1]), frame_number(depth_frames[-1])) + 1)):
        rgb = os.path.join(scene, 'color', f"{i:06}.jpg")
        depth = os.path.join(scene, 'depth', f"{i:06}.png")
        new_frames.append((rgb, depth))
        if not os.path.exists(rgb) or not os.path.exists(depth):
            print(f"warning: Missing rgb or depth frame {i:06}.")

    if len(new_frames) == 0:
        print("Nothing to cut")
        return
    os.makedirs(new_dir)
    copy_metadata(scene, new_dir)
    print(f"Moving other frames to {new_dir}")
    move_images(new_frames, new_dir)
    print("Done.")



if __name__ == "__main__":
    main()


