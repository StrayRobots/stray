
# Rotated bounding box detection using Detectron2

This tutorial will show you: 
1. How to produce rotated bounding box labels using labelme (<a href="https://github.com/wkentaro/labelme">https://github.com/wkentaro/labelme</a>) for custom datasets
2. How to configure and set up training for rotated bounding box detection using Detectron2
3. How to visualize predictions and labels

![Labelme polygon annotations](https://stray-data.nyc3.digitaloceanspaces.com/tutorials/labelmeui.png)
<p align = "center">
We are using labelme (<a href="https://github.com/wkentaro/labelme">https://github.com/wkentaro/labelme</a>) polygon annotation to label the container ships in the images (Photo by [Tom Fisk](https://www.pexels.com/@tomfisk) from [Pexels](https://www.pexels.com/)).
</p>


## Data labeling

### Install labelme
Labelme offers various options for installing the labeling GUI, please refer to the instructions here: <a href="https://github.com/wkentaro/labelme">https://github.com/wkentaro/labelme</a>

### Creating polygons
Start labelme with `labelme --autosave --nodata --keep-prev`. The GUI allows you to select the images to label one-by-one or based on a directory. It is highly recommended to place the images to label in a single directory, since a json file with the labels will be produced in the same location as the file.

The `--autosave` flag enables auto saving when moving from image to image The `--nodata` flag skips saving the actual image data in the json file that is produced for every image. Using `--keep-prev` can be considered optional, but it is very useful if the images are for example consecutive frames from a video since the option copies the labels from the previously labeled image to the current image.

You can create the polygon annotations with the "Create polygons" option in the GUI. The polygons will be used to determine the minimum-area rotated bounding box in the next step as a part of the model training. The logic to do this is provided out-of-the-box in the `stray` <a href="https://github.com/StrayRobots/stray">package</a> that is used in the example scripts.

This is an example of a json file that is produced for each image (in this case the file is called `1.json`):

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "ship",
      "points": [
        [
          239.0990099009901,
          420.2970297029703
        ],
        [
          423.25742574257424,
          338.1188118811881
        ],
        [
          444.54455445544556,
          345.049504950495
        ],
        [
          434.64356435643566,
          365.84158415841586
        ],
        [
          253.45544554455444,
          446.53465346534654
        ]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "1.png",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

## Installing Detectron2
Before we can visualize the annotations using visualization tools provided by Detectron2 and training the model, we need to install the package. Warning: this step may cause headaches.

### Install PyTorch, OpenCV, and Detectron2
Before installing Detectron2, we need to have PyTorch installed. This means that we can't provide a clean `requirements.txt` file with all the dependencies as there is no way to tell `pip` in which order to install the listed packages. Detectron also does not include the dependency in their install requirements for <a href="https://github.com/facebookresearch/detectron2/blob/9258799e4e72786edd67940872e0ed2c4387aac5/setup.py#L166">compatibility reasons</a>.

Depending on whether you want to use a CPU or a GPU (if available) with Detectron2, install the proper version from <a href="https://pytorch.org/">https://pytorch.org/</a>. The Detectron2 <a href=https://detectron2.readthedocs.io/en/latest/tutorials/install.html>installation documentation</a> also offers some background and debug steps if there are issues.

After installing `torch`, install Detectron2 using the instructions in the Detectron2 <a href=https://detectron2.readthedocs.io/en/latest/tutorials/install.html>installation documentation</a>.

Install rest of the dependencies with `pip install -r requirements.txt`.

### Test the installation and visualize the dataset

![Detectron2 rotated bounding boxes](https://stray-data.nyc3.digitaloceanspaces.com/tutorials/labels.png)
<p align = "center">
The polygons are used to determine the rotated bounding boxes. (Photo by [Tom Fisk](https://www.pexels.com/@tomfisk) from [Pexels](https://www.pexels.com/))
</p>

To see if everything works properly, you can run the visualization script (from stray/examples/detectron2) with `python visualize_dataset.py <path-to-dataset>` to visualize the annotations. As you can see the polygons are turned into rotated bounding boxes in the data loading step.

## Training the model

To run the training, run `python train_rotated_bbox.py <path-to-dataset> --num-gpus <gpus>`. The script and the `rotated_bbox_config.yaml` file contain various ways to configure the training, see the files for details. By default, the final and intermediate weights of the model are saved in the current working directory (`model_*.pth`).

## Predictions

![Detectron2 predictions](https://stray-data.nyc3.digitaloceanspaces.com/tutorials/predictions.png)
<p align = "center">
Predictions from the trained model. (Photo by [Tom Fisk](https://www.pexels.com/@tomfisk) from [Pexels](https://www.pexels.com/))
</p>

To visualize predictions from the trained model, run `python visualize_predictions <path-to-dataset> --weights <path-to-pth-model>`.