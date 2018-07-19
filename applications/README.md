## Applications

Applications is the Keras-Applications-like module for human rights violations recognition.
It provides model definitions and fine-tuned weights for a number of popular archictures, such as VGG16, VGG19, ResNet50 and VGG16-places365.

It also provides model definitions and fine-tuned weights for _CompoundNet_ a new approach introduced
in [_Exploring object-centric and scene-centric CNN features and their complementarity for human rights violations recognition in images_](https://arxiv.org/pdf/1805.04714.pdf)
paper, as well as a baseline model trained from scratch using the HRA dataset.

### Usage

All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the
image dimension ordering set in your Keras configuration file at ~/.keras/keras.json.
For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository will get built according to
the TensorFlow dimension ordering convention, "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (weights='HRA' argument in model constructor for
all models). Pre-trained weights using augmented samples can also be loaded upon
instantiation (data_augm_enabled = 'True' for all models). More than that, re-trained weights for dense layer only
(first phase _ConvNet as fixed feature extractor_) can also be automatically loaded upon instantiation
(weights='HRA' & mode= 'feature_extraction' arguments in model constructor for all models).
This can be useful when someone wants to fine-tune a model with different parameters,
(recall that in order to perform fine-tuning, all layers should start with properly trained weights).
Weights in all cases are automatically downloaded.  The input size used was 224x224 for all models.


### Available fine-tuned models
**Models for image classification with weights trained on HRA:**
- [VGG16](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/applications/hra_vgg16.py)
- [VGG19](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/applications/hra_vgg19.py)
- [ResNet50](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/applications/hra_resnet50.py)
- [VGG16-places365](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/applications/hra_vgg16_places365.py)
- [CompoundNet-VGG16](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/applications/compoundNet_vgg16.py)
- [CompoundNet-VGG19](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/applications/compoundNet_vgg19.py)
- [CompoundNet-ResNet50](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/blob/master/applications/compoundNet_resnet50.py)


### Examples - Classify HRA classes with VGG16

See `run_HRA_basic.py` and `run_HRA_unified.py`.

