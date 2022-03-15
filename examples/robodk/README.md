
# Box Picking with RoboDK

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
- `picking_setup.rdk`
  - This is the RoboDK simulation environment file.
- `model.pts`
  - A pretrained pytorch object detection model.

## Installing dependencies

We recommend using [Anaconda](https://docs.anaconda.com/anaconda/) for package management. To create a new environment and install the dependencies, run:
```
conda create -n robodk Python=3.8 && conda activate robodk
pip install -r requirements.txt
```

Follow the instructions [here](https://docs.strayrobots.io/toolkit/index.html) to install the Stray Robots Toolkit.

## Running the picking simulation with a trained model

First open the simulation in the RoboDK UI by opening RoboDK, then select File > Open... and open the `picking_setup.rdk` file.

To run the simulation with the included trained model, use the command `python pick.py --model model.pts`.

You should now see the simulation running with the robot picking boxes and moving them over to the other conveyor belt.

## Collecting a dataset and training an object detector

As is common these days, the object detection algorithm used is learning based. To train this algorithm, we need to collect example data from the robots workspace and annotate it to teach our robot to recognize the objects we want to pick.

To collect data, we provide the script `scan.py` which runs the simulation. It can be triggerred to stop the production line and scan the current state of the line. This is done by running the following command while the picking simulation is open in the RoboDK UI:
```
python scan.py --out boxes
```

To stop the line and scan the conveyor belt, press the `s` key.

The scans are saved in the path given to the `--out` parameter. For each performed scan, a subdirectory will be created within that directory which will contain the captured color and depth images, along with the camera poses.

After scanning, we need to process the scans to compute the 3D representation from the captured camera and depth images. This is done using the `stray studio integrate` command. Run it with:
```
stray studio integrate boxes/ --skip-mapping --voxel-size 0.005
```

As in this case, our camera is mounted on our calibrated robot, we use `--skip-mapping` parameter to tell the system that we know the camera poses and that these do not have to be inferred.

We then annotate each of the scans by opening the up in the Stray Studio user interface. A scan can be opened with:
```
stray studio open boxes/0001
```

In this case, we want to detect the top corners of each box. Therefore, we opt to annotate each scanned box with a rectangle annotation type. Here is what an annotated scan looks like:

![Annotated cardboard box in Stray Studio](https://stray-data.nyc3.digitaloceanspaces.com/tutorials/annotation.png)

