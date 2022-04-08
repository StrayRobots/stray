import click
from stray.debugger import VisualDebugger
from stray.scene import Scene
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation
import csv


def read_frames_csv(scene_path):
    path = os.path.join(scene_path, 'frames.csv')
    poses = []
    with open(path, 'rt') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            x, y, z, qx, qy, qz, qw = [float(f) for f in line[2:9]]
            pose = np.eye(4)
            pose[:3, 3] = [x, y, z]
            pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            poses.append(pose)
    return poses

@click.command()
@click.argument('scene-path', nargs=1)
@click.option('--cloud', is_flag=True)
@click.option('--mesh', is_flag=True)
def main(scene_path, cloud, mesh):
    scene = Scene(scene_path)
    debugger = VisualDebugger()

    #Origin
    debugger.add_frame(np.eye(4), 0.2)

    if cloud:
        pcl = o3d.io.read_point_cloud(os.path.join(scene_path, 'scene', 'cloud.ply'))
        debugger.add_mesh(pcl)
    if mesh:
        mesh = o3d.io.read_triangle_mesh(os.path.join(scene_path, 'scene', 'integrated.ply'))
        debugger.add_mesh(mesh)

    if os.path.exists(os.path.join(scene_path, 'frames.csv')):
        poses = read_frames_csv(scene_path)
    else:
        poses = scene.poses

    for pose in poses:
        debugger.add_frame(pose)

    debugger.show()

if __name__ == "__main__":
    main()
