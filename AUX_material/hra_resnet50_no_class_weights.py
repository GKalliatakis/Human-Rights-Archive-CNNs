# -*- coding: utf-8 -*-
"""Human Rights Archive (HRA) ResNet50 model for Keras

◾◾◾◾ Fine-tuning acc. ◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾
◾                                                     ◾
◾ ResNet50_avg  =>  25.55%                            ◾
◾                                                     ◾
◾ ResNet50_flatten  =>  30.00%                        ◾
◾                                                     ◾
◾ ResNet50_max  =>  25.55%                            ◾
◾                                                     ◾
◾◾◾◾ Fine-tuning coverage ◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾
◾                                                     ◾
◾ ResNet50_avg  =>  55%                               ◾
◾                                                     ◾
◾ ResNet50_flatten  =>  44%                           ◾
◾                                                     ◾
◾ ResNet50_max  =>  61%                               ◾
◾                                                     ◾
◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾

"""

from __future__ import division, print_function
import os

import warnings
import numpy as np

from keras import backend as K
from keras.utils.data_utils import get_file
from keras.layers import Input
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.optimizers import SGD

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------
# Weights obtained by re-training the last Dense layer
FEATURE_EXTRACTION_AVG_POOL_WEIGHTS_PATH = ''
FEATURE_EXTRACTION_AVG_POOL_fname = ''
FEATURE_EXTRACTION_FLATTEN_POOL_WEIGHTS_PATH = ''
FEATURE_EXTRACTION_FLATTEN_POOL_fname = ''
FEATURE_EXTRACTION_MAX_POOL_WEIGHTS_PATH = ''
FEATURE_EXTRACTION_MAX_POOL_fname = ''

AUGM_FEATURE_EXTRACTION_AVG_POOL_WEIGHTS_PATH = ''
AUGM_FEATURE_EXTRACTION_AVG_POOL_fname = ''
AUGM_FEATURE_EXTRACTION_FLATTEN_POOL_WEIGHTS_PATH = ''
AUGM_FEATURE_EXTRACTION_FLATTEN_POOL_fname = ''
AUGM_FEATURE_EXTRACTION_MAX_POOL_WEIGHTS_PATH = ''
AUGM_FEATURE_EXTRACTION_MAX_POOL_fname = ''

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------
# Weights obtained by un-freezing the two lower convolutional layers and retraining them
FINE_TUNING_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.1/fine_tuning_ResNet50_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_AVG_POOL_fname = 'fine_tuning_ResNet50_avg_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_FLATTEN_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.1/fine_tuning_ResNet50_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_FLATTEN_POOL_fname = 'fine_tuning_ResNet50_flatten_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.1/fine_tuning_ResNet50_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_MAX_POOL_fname = 'fine_tuning_ResNet50_max_pool_weights_tf_dim_ordering_tf_kernels'

AUGM_FINE_TUNING_AVG_POOL_WEIGHTS_PATH = ''
AUGM_FINE_TUNING_AVG_POOL_fname = ''
AUGM_FINE_TUNING_FLATTEN_POOL_WEIGHTS_PATH = ''
AUGM_FINE_TUNING_FLATTEN_POOL_fname = ''
AUGM_FINE_TUNING_MAX_POOL_WEIGHTS_PATH = ''
AUGM_FINE_TUNING_MAX_POOL_fname = ''

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------
# Weights obtained from removing the classification block from the fine-tuned models
FINE_TUNING_WEIGHTS_PATH_NO_TOP = ''
FINE_TUNING_WEIGHTS_PATH_NO_TOP_fname = ''
# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------



def HRA_ResNet50(include_top=True, weights='HRA',
                 input_tensor=None, input_shape=None,
                 mode= 'fine_tuning',
                 pooling_mode='avg',
                 classes=9,
                 data_augm_enabled=False):
    """Instantiates the ResNet50 architecture fine-tuned (2 steps) on Human Rights Archive dataset.

    Optionally loads weights pre-trained on Human Rights Archive Database.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
            'HRA' (pre-training on Human Rights Archive),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        mode: one of `TL` (transfer learning - freeze all but the penultimate layer and re-train the last Dense layer)
            or `FT` (fine-tuning - unfreeze the lower convolutional layers and retrain more layers) ,
        pooling_mode: Pooling mode that will be applied to the output of the last convolutional layer of the original model
            and thus the output of the model will be a 2D tensor.
            - `avg` means that global average pooling_mode operation for spatial data will be applied.
            - `max` means that global max pooling_mode operation for spatial data will be applied.
            - `flatten` means that the output of the the last convolutional
                layer of the original model will be flatten,
                resulting in a larger Dense layer afterwards.
        classes: optional number of classes to classify images into.
        data_augm_enabled: whether to use the augmented samples during training.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape
        """
    if not (weights in {'HRA', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `HRA` '
                         '(pre-training on Human Rights Archive), '
                         'or the path to the weights file to be loaded.')

    if not (mode in {'feature_extraction', 'fine_tuning'}):
        raise ValueError('The `mode` argument should be either '
                         '`feature_extraction` (freeze all but the penultimate layer and re-train the last Dense layer),'
                         'or `fine_tuning` (unfreeze the lower convolutional layers and retrain more layers). ')

    if not (pooling_mode in {'avg', 'max', 'flatten'}):
        raise ValueError('The `pooling_mode` argument should be either '
                         '`avg` (global average pooling_mode), `max` (global max pooling_mode)'
                         'or `flatten` (the output will be flatten). ')

    if mode == 'feature_extraction' and include_top is False:
        raise ValueError('The `include_top` argument can be set as false only '
                         'when the `mode` argument is `fine_tuning`. '
                         'If not, the returned model would have been literally the default '
                         'keras-applications model and not the one trained on HRA.')

    cache_subdir = 'HRA_models'

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

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input


    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=img_input)
    x = base_model.output

    # Classification block - build a classifier model to put on top of the convolutional model
    if include_top:

        # add a global spatial pooling_mode layer or flatten the obtained output from the original model
        if pooling_mode == 'avg':
            x = GlobalAveragePooling2D(name='GAP')(x)
        elif pooling_mode == 'max':
            x = GlobalMaxPooling2D(name='GMP')(x)
        elif pooling_mode == 'flatten':
            x = Flatten(name='FLATTEN')(x)

        # add a fully-connected layer
        x = Dense(256, activation='relu', name='FC1')(x)

        # When random init is enabled, we want to include Dropout,
        # otherwise when loading a pre-trained HRA model we want to omit that layer,
        # so the visualisations are done properly (there is an issue if it is included)
        if weights is None:
            x = Dropout(0.5,name='DROPOUT')(x)
        # and a logistic layer with the number of classes defined by the `classes` argument
        x = Dense(classes, activation='softmax', name='PREDICTIONS')(x)

    model = Model(inputs=inputs, outputs=x, name='HRA-ResNet50')


    # load weights
    if weights == 'HRA':
        if include_top:
            if mode == 'feature_extraction':
                for layer in base_model.layers:
                    layer.trainable = False

                model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                              loss='categorical_crossentropy')

                if data_augm_enabled:
                    if pooling_mode == 'avg':
                        weights_path = get_file(AUGM_FEATURE_EXTRACTION_AVG_POOL_fname,
                                                AUGM_FEATURE_EXTRACTION_AVG_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)
                    elif pooling_mode == 'flatten':
                        weights_path = get_file(AUGM_FEATURE_EXTRACTION_FLATTEN_POOL_fname,
                                                AUGM_FEATURE_EXTRACTION_FLATTEN_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)

                    elif pooling_mode == 'max':
                        weights_path = get_file(AUGM_FEATURE_EXTRACTION_MAX_POOL_fname,
                                                AUGM_FEATURE_EXTRACTION_MAX_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)

                else:
                    if pooling_mode == 'avg':
                        weights_path = get_file(FEATURE_EXTRACTION_AVG_POOL_fname,
                                                FEATURE_EXTRACTION_AVG_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)
                    elif pooling_mode == 'flatten':
                        weights_path = get_file(FEATURE_EXTRACTION_FLATTEN_POOL_fname,
                                                FEATURE_EXTRACTION_FLATTEN_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)


                    elif pooling_mode == 'max':
                        weights_path = get_file(FEATURE_EXTRACTION_MAX_POOL_fname,
                                                FEATURE_EXTRACTION_MAX_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)




            elif mode == 'fine_tuning':
                for layer in model.layers[:163]:
                    layer.trainable = False
                for layer in model.layers[163:]:
                    layer.trainable = True

                model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

                if data_augm_enabled:
                    if pooling_mode == 'avg':
                        weights_path = get_file(AUGM_FINE_TUNING_AVG_POOL_fname,
                                                AUGM_FINE_TUNING_AVG_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)
                    elif pooling_mode == 'flatten':
                        weights_path = get_file(AUGM_FINE_TUNING_FLATTEN_POOL_fname,
                                                AUGM_FINE_TUNING_FLATTEN_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)

                    elif pooling_mode == 'max':
                        weights_path = get_file(AUGM_FINE_TUNING_MAX_POOL_fname,
                                                AUGM_FINE_TUNING_MAX_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)

                else:
                    if pooling_mode == 'avg':
                        weights_path = get_file(FINE_TUNING_AVG_POOL_fname,
                                                FINE_TUNING_AVG_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)
                    elif pooling_mode == 'flatten':
                        weights_path = get_file(FINE_TUNING_FLATTEN_POOL_fname,
                                                FINE_TUNING_FLATTEN_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)


                    elif pooling_mode == 'max':
                        weights_path = get_file(FINE_TUNING_MAX_POOL_fname,
                                                FINE_TUNING_MAX_POOL_WEIGHTS_PATH,
                                                cache_subdir=cache_subdir)

        else:
            weights_path = get_file(FINE_TUNING_WEIGHTS_PATH_NO_TOP_fname,
                                    FINE_TUNING_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=cache_subdir)

        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)


    return model


if __name__ == '__main__':

    model = HRA_ResNet50(weights='HRA', mode='fine_tuning', pooling_mode='avg', include_top=True)

    model.summary()
