import os

def get_scene_paths(dataset_path):
    if os.path.exists(os.path.join(dataset_path, 'scene', 'integrated.ply')):
        scenes = [dataset_path]
    else:
        scenes = []
        for item in os.listdir(dataset_path):
            path = os.path.abspath(os.path.join(dataset_path, item))
            if os.path.isdir(path) and not item[0] == ".":
                scenes.append(path)
        scenes.sort()
    return scenes