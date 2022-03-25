import os
import click
from stray.segmentation import write_segmentation_masks

@click.command()
@click.argument('dataset', nargs=-1)
@click.option('--segmentation', default=False, is_flag=True,  help='Create segmentations.')
def main(**flags):
    scene_paths = [scene_path for scene_path in flags["dataset"] if os.path.isdir(scene_path)]
    if flags["segmentation"]:
        for scene_path in scene_paths:
            write_segmentation_masks(scene_path)

if __name__ == "__main__":
    main()