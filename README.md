# Release of Human-Rights-Archive-CNNs

We release various convolutional neural networks (CNNs) trained on _Human Rights Archive (HRA)_ to the public.
HRA is the first verified-by-experts repository of human rights violations photographs, labelled with human rights semantic categories,
comprising a list of the types of human rights abuses encountered at present.

## Data of Human Rights Archive

Here we release the data of Human Rights Archive to the public.
Human Rights Archive is the core set of HRA Database, which has been used to train the Human-Rights-Archive-CNNs.
We will add other kinds of annotation on the Human Rights Archive in the future.

There are 3050 train images from 8 human rights violations categories and one 'no violation' category in the HRA,
which are used to train the Human-Rights-Archive-CNNs.
The validation set can automatically be set in Keras by using the `validation_split argument` in `model.fit` accordingly.
There are 30 images per category in the testing set.

* [Train/Validation images](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/releases/download/v1.0/train_val.zip)
* [Test images](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/releases/download/v1.0/test.zip)

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



## Reference

Please cite the following [paper](https://arxiv.org/pdf/1805.04714.pdf) if you use the data or pre-trained CNN models.

```
 @article{kalliatakis2018exploring,
  title={Exploring object-centric and scene-centric CNN features and their complementarity for human rights violations recognition in images},
  author={Kalliatakis, Grigorios and Ehsan, Shoaib and Leonardis, Ales and McDonald-Maier, Klaus},
  journal={arXiv preprint arXiv:1805.04714},
  year={2018}
 }

```


Please email [Grigorios Kalliatakis](mailto:gkallia@essex.ac.uk) if you have any questions or comments.