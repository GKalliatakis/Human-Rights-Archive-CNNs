# -*- coding: utf-8 -*-
"""Simple (baseline) deep CNN on the Human Rights Archive image dataset.
The baseline model is a simple stack of 3 convolution layers with a ReLU activation
and followed by max-pooling layers. This is very similar to the architectures that
Yann LeCun advocated in the 1990s for image classification (with the exception of ReLU).


◾◾◾◾ Baseline mode ◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾
◾                                                                       ◾
◾ baseline_10epochs  =>  0.137037037037                                 ◾
◾                                                                       ◾
◾ baseline_20epochs  =>  0.155555555556                                 ◾
◾                                                                       ◾
◾ baseline_40epochs  =>  0.144444444444                                 ◾
◾                                                                       ◾
◾                                                                       ◾
◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾


# References
- [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]

"""

import os

import keras.backend as K

from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Model
from keras.applications.imagenet_utils import _obtain_input_shape


WEIGHTS_PATH_10_EPOCHS = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.5/baseline_model_10_epochs_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_20_EPOCHS = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.5/baseline_model_20_epochs_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_40_EPOCHS = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.5/baseline_model_40_epochs_weights_tf_dim_ordering_tf_kernels.h5'





def baseline_model(include_top=True, weights='HRA',
                   input_tensor=None, input_shape=None,
                   classes=9,
                   epochs=40):
    """Instantiates a baseline deep CNN architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    # Arguments
        include_top: whether to include the 2 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'HRA' (pre-training on Human Rights Archive),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        epochs: Integer. Number of epochs to train the model.
        augmented_samples: whether to use the augmented samples during training

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.

    """

    if not (weights in {'HRA', None} or os.path.exists(weights)):
        raise ValueError('The `weights_top_layers` argument should be either '
                         '`None` (random initialization), `HRA` '
                         '(pre-training on Human Rights Archive), '
                         'or the path to the weights file to be loaded.')

    if weights == 'HRA' and include_top and classes != 9:
        raise ValueError('If using `weights` as Human Rights Archive, `classes` should be 9.')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Define The Neural Network Model

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # the model so far outputs 3D feature maps (height, width, features)

    # On top of it we stick two fully-connected layers. We end the model with 9 final outputs representing probabilities for HRA classes
    # and a softmax activation, which is perfect for a multi-class classification.
    # To go with it we will also use the categorical_crossentropy loss to train our model.
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(64, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='HRA-baseline_convnet')

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()


    # load weights
    if weights == 'HRA' and epochs == 10:
        if include_top:
            weights_path = get_file('baseline_model_10_epochs_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_10_EPOCHS,
                                    cache_subdir='hra_models')

    elif weights == 'HRA' and epochs == 20:
        if include_top:
            weights_path = get_file('baseline_model_20_epochs_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_20_EPOCHS,
                                    cache_subdir='hra_models')

    elif weights == 'HRA' and epochs == 40:
        if include_top:
            weights_path = get_file('baseline_model_40_epochs_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_40_EPOCHS,
                                    cache_subdir='hra_models')

    model.load_weights(weights_path)

    return model


if __name__=="__main__":

    model = baseline_model(classes=9, epochs=40, weights='HRA')
    model.summary()

