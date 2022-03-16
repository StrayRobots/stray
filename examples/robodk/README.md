
# Automatic pick and place with computer vision and RoboDK

<a target="_blank" href="https://stray-data.nyc3.digitaloceanspaces.com/tutorials/pick_video.mp4" title="Picking with RoboDK"><img src="https://stray-data.nyc3.digitaloceanspaces.com/tutorials/pick_cover.png" alt="Picking with RoboD" /></a>
<p align = "center">
The complete pick and place example.
</p>

This example shows how to build datasets with the [Stray Robots Toolkit](https://docs.strayrobots.io) for picking and placing cardboard boxes of variable sizes within a [RoboDK](https://robodk.com/) simulation environment.

The purpose of this simulation is to showcase how computer vision can be used in dynamic pick and place applications. In this case, our robot is tasked to move boxes rolling on one conveyor belt to the other.

The robot waits for boxes to come down the conveyor belt, where it detects the top corners of the box. We then compute a picking location at the center top of the cardboard box and command the robot to pick it up with its suction cup gripper. The robot then moves the box over to the other conveyor belt and starts over.

## Project structure

The main files in this project are:
- `pick.py`
  - Contains the main script containing the picking logic.
- `scan.py`
  - This script is used to collect datasets for training the object detector.
- `detect.py`
  - Contains the object detection logic.
- `simulation.py`
  - Contains the simulation logic for the conveyor belt, reseting and spawning boxes etc.
- `train.py`
  - Contains the logic for training a detection model.
- `convert_model.py`
  - Contains the logic for converting a <a href="https://pytorch-lightning.readthedocs.io/en/latest/">PyTorch Lightning</a> checkpoint into a serialized model that can be run in production.
- `picking_setup.rdk`
  - This is the RoboDK simulation environment file.
- `model.pt`
  - A pretrained pytorch object detection model.

## Installing dependencies

We recommend using [Anaconda](https://docs.anaconda.com/anaconda/) for package management. To create a new environment and install the dependencies, run:
```
conda create -n robodk python=3.8 && conda activate robodk
pip install -r requirements.txt
```

Follow the instructions [here](https://docs.strayrobots.io/toolkit/index.html) to install the Stray Robots Toolkit.

## Collecting a dataset and training an object detector

As is common these days, the object detection algorithm used is learning based. To train this algorithm, we need to collect example data from the robot's workspace and annotate it to teach our robot to recognize the objects we want to pick. The rest of this section will show you how to collect a custom dataset. Alternatively you can <a href="https://stray-data.nyc3.digitaloceanspaces.com/tutorials/boxes.zip">download</a> a sample dataset to proceed to model training and the next section. The <a href="https://stray-data.nyc3.digitaloceanspaces.com/tutorials/boxes.zip">sample</a> includes scans of 20 boxes.

First open the simulation in the RoboDK UI by opening RoboDK, then select File > Open... and open the `picking_setup.rdk` file.
To collect data, we provide the script `scan.py` which runs the simulation. It can be triggerred to stop the production line and scan the current state of the line. This is done by running the following command while the picking simulation is open in the RoboDK UI:
```
python scan.py <out>
```

To stop the line and scan the conveyor belt, press the `s` key.

The scans are saved in the path given by `out`. For each performed scan, a subdirectory will be created within that directory which will contain the captured color and depth images, along with the camera poses.

After scanning, we need to process the scans to compute the 3D representation from the captured camera and depth images. This is done using the `stray integrate` command. Run it with:
```
stray integrate <out> --skip-mapping --voxel-size 0.005
```

As in this case, our camera is mounted on our calibrated robot, we use `--skip-mapping` parameter to tell the system that we know the camera poses and that these do not have to be inferred.

We then annotate each of the scans by opening the up in the Stray Studio user interface. A scan can be opened with:
```
stray studio boxes/0001
```

In this case, we want to detect the top corners of each box. Therefore, we opt to annotate each scanned box with a rectangle annotation type. Here is what an annotated scan looks like:

![Annotated cardboard box in Stray Studio](https://stray-data.nyc3.digitaloceanspaces.com/tutorials/annotation.png)

## Training a picking model
Once the dataset is collected (or downloading the <a href="https://stray-data.nyc3.digitaloceanspaces.com/tutorials/boxes.zip">sample</a>) we can go ahead and train the model. Alternatively you can also use the pretrained <a href="https://github.com/StrayRobots/stray/blob/main/examples/robodk/model.pt">model</a> and proceed to the next section.

The model training can be run with

```sh
python train.py <out> --eval-folder <path-to-eval-data> --num-epochs <num-epochs> --primitive rectangle --num-instances 1
```

The main available paramaters are:
- `out`
  - Path to the directory containing the scans
- `--eval-folder`
  - Path to the directory containing evaluation scans, for testing purposes this can be the same as `out`
- `--num-epochs`
  - For how many iterations the model should be trained. 100 is enough for the example dataset of 20 scans.
- `--primitive`
  - Which primitive type to use for determining the keypoints on the box. In this example we use `rectangle`.
- `--num-instances`
  - How many instances of `primitive` should be detected. The instances should be labeled with unique instance ids in Studio, ranging from `0` to `num-instances - 1`. In this case there is only one instance per scene/image.
- `--batch-size (default 4)`
  - Batch size to use during training, adjust this as high as possible as long as there are no memory errors

For additional settings refer to the `train.py` file.

The training is implemented using <a href="https://pytorch-lightning.readthedocs.io/en/latest/">PyTorch Lightning</a>. Logs of the training are saved to `./train_logs` but this can be adjusted with the `--logs` flag. Intermediate versions of the model are saved to `./train_logs/default/version_x/checkpoints`.

Once the training is completed, we can pick one of the checkpoints from `./train_logs/default/version_x/checkpoints` and convert it into a serialized model that can be used in production and the example picking script.

The model can be converted with `python convert_model.py <path-to-checkpoint>` and it will be saved as `model.pt`.

## Running the picking simulation with the trained model

Again, open the simulation in the RoboDK UI by opening RoboDK, then select File > Open... and open the `picking_setup.rdk` file.

To run the simulation with the trained model, use the command `python pick.py --model model.pt`.

<a target="_blank" href="https://stray-data.nyc3.digitaloceanspaces.com/tutorials/pick_video.mp4" title="Picking with RoboDK"><img src="https://stray-data.nyc3.digitaloceanspaces.com/tutorials/pick_cover.png" alt="Picking with RoboD" /></a>
<p align = "center">
You should now see the simulation running with the robot picking boxes and moving them over to the other conveyor belt.
</p>
