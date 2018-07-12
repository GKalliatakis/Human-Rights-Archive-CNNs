# Release of Human-Rights-Archive-CNNs

We release various convolutional neural networks (CNNs) trained on _Human Rights Archive (HRA)_ to the public.
HRA is the first verified-by-experts repository of human rights violations photographs, labelled with human rights semantic categories,
comprising a list of the types of human rights abuses encountered at present.

## Pre-trained CNN models on HRA (`applications` module):

* Baseline model
* ResNet50-HRA
* VGG16-HRA
* VGG16-Places365-HRA
* VGG19
* CompoundNet-ResNet50
* CompoundNet-VGG16
* CompoundNet-VGG19


**Train Human-Rights-Archive-CNNs using Keras. The training script is [here](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/train_HRA.py).**

    Example
    --------
    >>> python train_HRA.py --pre_trained_model VGG16 --pooling_mode avg


## Performance of the Human-Rights-Archive-CNNs

The performance of the CNNs is listed below.

<p align="center">
  <img src="https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/table3.png?raw=true"/>
</p>


Some qualitative prediction results using the VGG16-Places365:

<p align="center">
  <img src="https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/fig8.png?raw=true"/>
</p>