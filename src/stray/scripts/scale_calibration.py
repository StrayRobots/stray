import os
import click
import numpy as np
import json

def _resize_camera_matrix(camera_matrix, scale_x, scale_y):
    if scale_x == 1.0 and scale_y == 1.0:
        return camera_matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return np.array([[fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]])

def modify_intrinsics(data, width, height):
    old_width, old_height = data['width'], data['height']
    intrinsic_matrix = np.array(data['intrinsic_matrix']).reshape(3, 3).T
    intrinsics_scaled = _resize_camera_matrix(intrinsic_matrix, width / old_width, height / old_height)

    data['intrinsic_matrix'] = [intrinsics_scaled[0, 0], 0.0, 0.0,
            0.0, intrinsics_scaled[1, 1], 0.0,
            intrinsics_scaled[0, 2], intrinsics_scaled[1, 2], 1.0]
    data['width'] = width
    data['height'] = height
    return data

@click.command()
@click.argument('scenes', nargs=-1)
@click.option('--width', '-w', type=int, help="New width", default=640)
@click.option('--height', '-h', type=int, help="New height", default=480)
def main(scenes, width, height):
    """
    Command for scaling the camera_intrinsics.json configuration for a different image size.

    Usage: scale <stray-scenes> --width <new-width> --height <new-height>

    The size the camera was calibrated for will be inferred from the existing camera_intrinsics.json file.
    """
    if len(scenes) == 0:
        print("No scenes provided.")
        exit(1)

    for scene in scenes:
        print(f"Scaling {scene}")
        with open(os.path.join(scene, 'camera_intrinsics.json'), 'rt') as f:
            intrinsic_data = json.load(f)

        new_data = modify_intrinsics(intrinsic_data, width, height)

        with open(os.path.join(scene, 'camera_intrinsics.json'), 'wt') as f:
            f.write(json.dumps(new_data, indent=4, sort_keys=True))

    print("Done.")

if __name__ == "__main__":
    main()


