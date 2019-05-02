# Release of Human-Rights-Archive-CNNs [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=The%20Human-Rights-Archive-CNNs%20for%20Human%20Rights%20Violations%20Recognition&url=https://github.com/GKalliatakis/Human-Rights-Archive-CNNs&hashtags=ML,DeepLearning,CNNs,HumanRights,HumanRightsViolations)

We release various convolutional neural networks (CNNs) trained on _Human Rights Archive (HRA)_ to the public.
HRA is the first verified-by-experts repository of human rights violations photographs, labelled with human rights semantic categories,
comprising a list of the types of human rights abuses encountered at present.

### Data of Human Rights Archive

Here we release the data of Human Rights Archive to the public.
Human Rights Archive is the core set of HRA Database, which has been used to train the Human-Rights-Archive-CNNs.
We will add other kinds of annotation on the Human Rights Archive in the future.

There are 3050 train images from 8 human rights violations categories and one 'no violation' category in the HRA,
which are used to train the Human-Rights-Archive-CNNs.
The validation set can automatically be set in Keras by using the `validation_split argument` in `model.fit` accordingly.
There are 30 images per category in the testing set.

* [Train/Validation images](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/releases/download/v1.0/train_val.zip)
* [Test images](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/releases/download/v1.0/test.zip)

### Models for image classification with weights trained on HRA (`applications` module):

Pre-trained weights can be automatically loaded upon instantiation
(weights='HRA' argument in model constructor for all models).
Weights are automatically downloaded if necessary, and cached locally in `~/.keras/HRA_models/` just like the default `applications` module of
the Keras deep learning library.

All architectures are compatible with both TensorFlow and Theano,
and upon instantiation the models will be built according to the
image dimension ordering set in your Keras configuration file at ~/.keras/keras.json.
For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository
will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".

This repository contains code for the following fine-tuned Keras models:

* Baseline model (very similar to the architectures that Yann LeCun advocated in the 1990s)
* ResNet50
* VGG16
* VGG16-Places365
* VGG19
* CompoundNet-ResNet50
* CompoundNet-VGG16
* CompoundNet-VGG19


### Git

To download the very latest source from the Git server do this:

    git clone https://github.com/GKalliatakis/Human-Rights-Archive-CNNs.git

(you'll get a directory named Human-Rights-Archive-CNNs created, filled with the source code)

### Usage

**Train Human-Rights-Archive-CNNs using Keras. The training script is [here](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/train_HRA.py).**

    Example
    --------
    >>> python train_HRA.py --pre_trained_model VGG16 --pooling_mode avg


**Run [single image inference code](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/run_HRA_basic.py) to get the human rights violations predictions from Human-Rights-Archive-CNNs**

    Example
    --------
    >>> python run_HRA_basic.py --img_path path/to/your/image/xxxx.jpg --pre_trained_model VGG16 --pooling_mode avg

**or [run unified code](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/run_HRA_unified.py) to predict human rights violations categories,
and the class activation map together from Human-Rights-Archive-CNNs**

    Example
    --------
    >>> python run_HRA_unified.py --img_path path/to/your/image/xxx.jpg --pre_trained_model VGG16 --pooling_mode avg --to_file output_filename.png


### Dependencies

* Keras 2.1.5 or above
* OpenCV (relevant only to `run_HRA_unified.py` where we generate an image that superimposes the original image with the class activation heatmap)



### Performance of the Human-Rights-Archive-CNNs

The performance of the CNNs is listed below.

<p align="center">

|                 |   Pool  | Top-1 acc. | Coverage | Trainable Params. |
|:---------------:|:-------:|:----------:|:--------:|:-----------------:|
|      VGG16      |         |   34.44%   |    45%   |     4,853,257     |
|      VGG19      |   avg   | **35.18%** |    42%   |     4,853,257     |
|     ResNet50    |         |   25.55%   |    55%   |     4,992,521     |
| VGG16-places365 |         |   30.00%   |    32%   |     4,853,257     |
|                 |         |            |          |                   |
|      VGG16      |         |   31.85%   |    55%   |     8,784,905     |
|      VGG19      | flatten |   31.11%   |    50%   |     8,784,905     |
|     ResNet50    |         |   30.00%   |    44%   |     4,992,521     |
| VGG16-places365 |         |   28.51%   |    52%   |     8,784,905     |
|                 |         |            |          |                   |
|      VGG16      |         |   28.14%   |  **64%** |     4,853,257     |
|      VGG19      |   max   |   29.62%   |    61%   |     4,853,257     |
|     ResNet50    |         |   25.55%   |    61%   |     4,992,521     |
| VGG16-places365 |         |   26.66%   |    51%   |     4,853,257     |

</p>


**Some qualitative prediction results using the VGG16-Places365:**

<p align="center">
  <img src="https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/fig8.png?raw=true"/>
</p>



### Human rights violations recognition demo
**Does your image depict a human right violation? Upload to find out!**

The application can be executed on a PC without Internet access or it can be installed on a remote server,
where you can access it through the Internet.

Run the following command to open up the application:

    Example
    --------
    >>> python Human-Rights-Archive-CNNs/web_app/app.py


Then you'll see the application opening in the web browser on the following address: http://0.0.0.0:5000/.

---


### Reference

Please cite the following [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8606079) if you use the data or pre-trained CNN models.

```
@article{kalliatakis2019exploring,
  title={Exploring object-centric and scene-centric CNN features and their complementarity for human rights violations recognition in images},
  author={Kalliatakis, Grigorios and Ehsan, Shoaib and Leonardis, Ale{\v{s}} and Fasli, Maria and McDonald-Maier, Klaus D},
  journal={IEEE Access},
  volume={7},
  pages={10045--10056},
  year={2019},
  publisher={IEEE}
}

```


Please email [Grigorios Kalliatakis](mailto:gkallia@essex.ac.uk) if you have any questions or comments.