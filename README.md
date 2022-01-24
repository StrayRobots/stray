# Stray SDK
Stray SDK is the primary way for integrating Stray Robots capabilities into your app. The SDK supports visualizing data, training models, and integrating detectors into custom solutions.

## Installation
We recommend using a virtual environment such as [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). The environment should have `python>=3.8`.

To install the package, run `pip install "git+https://github.com/StrayRobots/stray"`

## Display a dataset
The scenes should be in the Stray scene format. See details: https://docs.strayrobots.io/formats/data.html

Run `stray-show <scene_1> <scene_2> <...> <--flags>`

Available flags are:
* `--bbox` show 2D bounding boxes (assuming annotations exist, see https://docs.strayrobots.io/commands/studio.html)
* `--keypoints`show keypoints (assuming annotations exist)
* `--segmentation` show object segmentations (assuming annotations exist)
* `--save` save the displayed images into the scene directory
* `--rate <RATE>` how many frames to show per second
* `--scale <SCALE>` scale the original images when displaying (to fit large images on the screen)

To pause, press `space`. To skip a scene, press `s`. To quit the dataset preview, press `q`.

## Train a model
The scenes should be in the Stray scene format. See details: https://docs.strayrobots.io/formats/data.html

Run `stray-train <model-type> --train-data <TRAIN-PATH> --eval-data <EVAL-path> <--additional-flags>`

Currently the only option for `<model-type>` is `keypoint-heatmap`

`<TRAIN-PATH>` and `<EVAL-PATH>` directories should include the Stray scenes to use during training/eval.

The progress of the training is saved into `./lightning_logs` relative to the current directory. We use [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) for implementing the training procedure. The progress can be viewed using `tensorboard --logdir <PATH-TO-LOGS>` (assuming you have installed [Tensorboard](https://www.tensorflow.org/tensorboard)).

The additional flags are:
* `--batch-size <BATCH-SIZE>` batch size to use during training
* `--num-workers <NUM-WORKERS>` number of workers to use during data loading
* `--num-epochs <NUM-EPOCHS>` number of epochs to train for
* `--width <WIDTH>` image width to use during training
* `--height <HEIGHT>` image height to use during training
* `--num-heatmaps <NUM-HEATMAPS>` number of heatmaps to output (only applies to heatmap-based training)
* `--lr <LR>` learning rate
* `--fp16` use 16-bit precision
* `--tune` tune hyperparameters prior to training
* `--checkpoint <CHECKPOINT>` path to a Pytorch Lightning checkpoint to continue a previous training
