# -*- coding: utf-8 -*-
"""Uses CompoundNet in order to combine features extracted from VGG19 and VGG16-Places365 using the (complete with 9 classes) HRA dataset.

◾◾◾◾ Fine-tuning ◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾
◾                                                                                                             ◾
◾ VGG19_fine_tuning_avg_pool  =>  0.351851851852                                                              ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_average_fusion_avg_pool  =>  0.27037037037                                    ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_concatenate_fusion_avg_pool  =>  0.277777777778                               ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_maximum_fusion_avg_pool  =>  0.288888888889                                   ◾
◾ ------------------------------------------------------------------------------------------------------------◾
◾ VGG19_fine_tuning_flatten  =>  0.311111111111                                                               ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_average_fusion_flatten  =>  0.255555555556                                    ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_concatenate_fusion_flatten  =>  0.288888888889                                ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_maximum_fusion_flatten  =>  0.281481481481                                    ◾
◾ ------------------------------------------------------------------------------------------------------------◾
◾ VGG19_fine_tuning_max_pool  =>  0.296296296296                                                              ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_average_fusion_max_pool  =>  0.27037037037                                    ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_concatenate_fusion_max_pool  =>  0.262962962963                               ◾
◾                                                                                                             ◾
◾ CompoundNet_VGG19_fine_tuning_maximum_fusion_max_pool  =>  0.262962962963                                   ◾
◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾◾

"""
import os
import sys

from keras.utils.data_utils import get_file
from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, GlobalAveragePooling2D,GlobalMaxPooling2D
from keras import backend as K
from keras.layers.merge import concatenate,average, maximum
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

from applications.vgg16_places_365 import VGG16_Places365

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------
# Weights obtained by re-training the last Dense layer
FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_fname = 'compoundNet_feature_extraction_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_fname = 'compoundNet_feature_extraction_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_fname = 'compoundNet_feature_extraction_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_fname = 'compoundNet_feature_extraction_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_fname = 'compoundNet_feature_extraction_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_fname = 'compoundNet_feature_extraction_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_fname = 'compoundNet_feature_extraction_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_fname = 'compoundNet_feature_extraction_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/compoundNet_feature_extraction_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_fname = 'compoundNet_feature_extraction_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'


AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_fname = 'augm_compoundNet_feature_extraction_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_fname = 'augm_compoundNet_feature_extraction_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_fname = 'augm_compoundNet_feature_extraction_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_fname = 'augm_compoundNet_feature_extraction_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_fname = 'augm_compoundNet_feature_extraction_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_fname = 'augm_compoundNet_feature_extraction_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_fname = 'augm_compoundNet_feature_extraction_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_fname = 'augm_compoundNet_feature_extraction_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.3/augm_compoundNet_feature_extraction_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_fname = 'augm_compoundNet_feature_extraction_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'


# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------

# Weights obtained by un-freezing the two lower convolutional layers and retraining them
FINE_TUNING_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_AVERAGE_FUSION_AVG_POOL_fname = 'compoundNet_fine_tuning_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_AVERAGE_FUSION_FLATTEN_fname = 'compoundNet_fine_tuning_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_AVERAGE_FUSION_MAX_POOL_fname = 'compoundNet_fine_tuning_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_fname = 'compoundNet_fine_tuning_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_CONCATENATE_FUSION_FLATTEN_fname = 'compoundNet_fine_tuning_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_fname = 'compoundNet_fine_tuning_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_fname = 'compoundNet_fine_tuning_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_MAXIMUM_FUSION_FLATTEN_fname = 'compoundNet_fine_tuning_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/compoundNet_fine_tuning_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_fname = 'compoundNet_fine_tuning_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'


AUGM_FINE_TUNING_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_AVERAGE_FUSION_AVG_POOL_fname = 'augm_compoundNet_fine_tuning_VGG19_average_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FINE_TUNING_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_AVERAGE_FUSION_FLATTEN_fname = 'augm_compoundNet_fine_tuning_VGG19_average_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FINE_TUNING_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_AVERAGE_FUSION_MAX_POOL_fname = 'augm_compoundNet_fine_tuning_VGG19_average_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AUGM_FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_fname = 'augm_compoundNet_fine_tuning_VGG19_concatenate_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FINE_TUNING_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_CONCATENATE_FUSION_FLATTEN_fname = 'augm_compoundNet_fine_tuning_VGG19_concatenate_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_fname = 'augm_compoundNet_fine_tuning_VGG19_concatenate_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
AUGM_FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_fname = 'augm_compoundNet_fine_tuning_VGG19_maximum_fusion_avg_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FINE_TUNING_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_MAXIMUM_FUSION_FLATTEN_fname = 'augm_compoundNet_fine_tuning_VGG19_maximum_fusion_flatten_pool_weights_tf_dim_ordering_tf_kernels'
AUGM_FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/0.4/augm_compoundNet_fine_tuning_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels.h5'
AUGM_FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_fname = 'augm_compoundNet_fine_tuning_VGG19_maximum_fusion_max_pool_weights_tf_dim_ordering_tf_kernels'

# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------

# Weights obtained from removing the classification block from the fine-tuned models
FINE_TUNING_AVERAGE_FUSION_WEIGHTS_PATH_NO_TOP = ''
FINE_TUNING_AVERAGE_FUSION_NO_TOP_fname =''
FINE_TUNING_CONCATENATE_FUSION_WEIGHTS_PATH_NO_TOP = ''
FINE_TUNING_CONCATENATE_FUSION_NO_TOP_fname= ''
FINE_TUNING_MAXIMUM_FUSION_WEIGHTS_PATH_NO_TOP = ''
FINE_TUNING_MAXIMUM_FUSION_NO_TOP_fname =''
# ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------  ======================  ------------------------

def CompoundNet_VGG19(include_top=True, weights=None,
                      input_tensor=None, input_shape=None,
                      fusion_strategy='concatenate', mode= 'fine_tuning',
                      pooling_mode ='avg',
                      classes=9,
                      data_augm_enabled=False):
    """Instantiates the CompoundNet VGG19 architecture fine-tuned (2 steps) on Human Rights Archive dataset.

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
            fusion_strategy: one of `concatenate` (feature vectors of different sources are concatenated into one super-vector),
                `average` (the feature set is averaged)
                or `maximum` (selects the highest value from the corresponding features).
            mode: one of `feature_extraction` (freeze all but the penultimate layer and re-train the last Dense layer)
                or `fine_tuning` (unfreeze the lower convolutional layers and retrain more layers).
            pooling_mode: Optional pooling_mode mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling_mode
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling_mode will
                    be applied.
            classes: optional number of classes to classify images into,
                                only to be specified if `weights` argument is `None`.
            data_augm_enabled: whether to use the augmented samples during training.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`.
        """

    if not (weights in {'HRA', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `HRA` '
                         '(pre-training on Human Rights Archive), '
                         'or the path to the weights file to be loaded.')

    if not (fusion_strategy in {'concatenate', 'average', 'maximum'}):
        raise ValueError('The `fusion_strategy` argument should be either '
                         '`concatenate` (feature vectors of different sources are concatenated into one super-vector), '
                         '`average` (the feature set is averaged) '
                         'or `maximum` (selects the highest value from the corresponding features).')

    if not (pooling_mode in {'avg', 'max', 'flatten'}):
        raise ValueError('The `pooling_mode` argument should be either '
                         '`avg` (GlobalAveragePooling2D), `max` '
                         '(GlobalMaxPooling2D), '
                         'or `flatten` (Flatten).')


    if weights == 'HRA' and classes != 9:
        raise ValueError('If using `weights` as Human Rights Archive, `classes` should be 9.')

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


    input_tensor = Input(shape=(224, 224, 3))

    object_centric_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

    scene_centric_model = VGG16_Places365(input_tensor=input_tensor,weights='places', include_top=False)


    # retrieve the ouputs
    object_model_output = object_centric_model.output
    scene_model_output = scene_centric_model.output

    # We will feed the extracted features to a merging layer
    if fusion_strategy == 'concatenate':
        merged = concatenate([object_model_output, scene_model_output])

    elif fusion_strategy == 'average':
        merged = average([object_model_output, scene_model_output])

    else:
        merged = maximum([object_model_output, scene_model_output])

    if include_top:
        if pooling_mode == 'avg':
            x = GlobalAveragePooling2D(name='GAP')(merged)
        elif pooling_mode == 'max':
            x = GlobalMaxPooling2D(name='GMP')(merged)
        elif pooling_mode == 'flatten':
            x = Flatten(name='FLATTEN')(merged)

        x = Dense(256, activation='relu', name='FC1')(x)  # let's add a fully-connected layer

        # When random init is enabled, we want to include Dropout,
        # otherwise when loading a pre-trained HRA model we want to omit
        # Dropout layer so the visualisations are done properly (there is an issue if it is included)
        if weights is None:
            x = Dropout(0.5, name='DROPOUT')(x)
        # and a logistic layer with the number of classes defined by the `classes` argument
        x = Dense(classes, activation='softmax', name='PREDICTIONS')(x)  # new softmax layer

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # this is the transfer learning model we will train
    model = Model(inputs=inputs, outputs=x, name='CompoundNet-VGG19')


    # load weights
    if weights == 'HRA':
        if include_top:
            if mode == 'feature_extraction':
                for layer in object_centric_model.layers:
                    layer.trainable = False

                for layer in scene_centric_model.layers:
                    layer.trainable = False

                model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                              loss='categorical_crossentropy')

                if data_augm_enabled:

                    if fusion_strategy == 'concatenate':
                        if pooling_mode == 'avg':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_fname,
                                                    AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_fname,
                                                    AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_fname,
                                                    AUGM_FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)


                    elif fusion_strategy == 'average':
                        if pooling_mode == 'avg':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_fname,
                                                    AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_fname,
                                                    AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_fname,
                                                    AUGM_FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                    elif fusion_strategy == 'maximum':
                        if pooling_mode == 'avg':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_fname,
                                                    AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_fname,
                                                    AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_fname,
                                                    AUGM_FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)
                else:
                    if fusion_strategy == 'concatenate':
                        if pooling_mode == 'avg':
                            weights_path = get_file(FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_fname,
                                                    FEATURE_EXTRACTION_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_fname,
                                                    FEATURE_EXTRACTION_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_fname,
                                                    FEATURE_EXTRACTION_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)


                    elif fusion_strategy == 'average':
                        if pooling_mode == 'avg':
                            weights_path = get_file(FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_fname,
                                                    FEATURE_EXTRACTION_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_fname,
                                                    FEATURE_EXTRACTION_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_fname,
                                                    FEATURE_EXTRACTION_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                    elif fusion_strategy == 'maximum':
                        if pooling_mode == 'avg':
                            weights_path = get_file(FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_fname,
                                                    FEATURE_EXTRACTION_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_fname,
                                                    FEATURE_EXTRACTION_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_fname,
                                                    FEATURE_EXTRACTION_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)


            elif mode == 'fine_tuning':
                for layer in model.layers[:36]:
                    layer.trainable = False
                for layer in model.layers[36:]:
                    layer.trainable = True

                model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                              loss='categorical_crossentropy')

                if data_augm_enabled:

                    if fusion_strategy == 'concatenate':
                        if pooling_mode == 'avg':
                            weights_path = get_file(AUGM_FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_fname,
                                                    AUGM_FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(AUGM_FINE_TUNING_CONCATENATE_FUSION_FLATTEN_fname,
                                                    AUGM_FINE_TUNING_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(AUGM_FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_fname,
                                                    AUGM_FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)


                    elif fusion_strategy == 'average':
                        if pooling_mode == 'avg':
                            weights_path = get_file(AUGM_FINE_TUNING_AVERAGE_FUSION_AVG_POOL_fname,
                                                    AUGM_FINE_TUNING_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(AUGM_FINE_TUNING_AVERAGE_FUSION_FLATTEN_fname,
                                                    AUGM_FINE_TUNING_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(AUGM_FINE_TUNING_AVERAGE_FUSION_MAX_POOL_fname,
                                                    AUGM_FINE_TUNING_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                    elif fusion_strategy == 'maximum':
                        if pooling_mode == 'avg':
                            weights_path = get_file(AUGM_FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_fname,
                                                    AUGM_FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(AUGM_FINE_TUNING_MAXIMUM_FUSION_FLATTEN_fname,
                                                    AUGM_FINE_TUNING_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(AUGM_FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_fname,
                                                    AUGM_FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)
                else:

                    if fusion_strategy == 'concatenate':
                        if pooling_mode == 'avg':
                            weights_path = get_file(FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_fname,
                                                    FINE_TUNING_CONCATENATE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(FINE_TUNING_CONCATENATE_FUSION_FLATTEN_fname,
                                                    FINE_TUNING_CONCATENATE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_fname,
                                                    FINE_TUNING_CONCATENATE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)


                    elif fusion_strategy == 'average':
                        if pooling_mode == 'avg':
                            weights_path = get_file(FINE_TUNING_AVERAGE_FUSION_AVG_POOL_fname,
                                                    FINE_TUNING_AVERAGE_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(FINE_TUNING_AVERAGE_FUSION_FLATTEN_fname,
                                                    FINE_TUNING_AVERAGE_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(FINE_TUNING_AVERAGE_FUSION_MAX_POOL_fname,
                                                    FINE_TUNING_AVERAGE_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                    elif fusion_strategy == 'maximum':
                        if pooling_mode == 'avg':
                            weights_path = get_file(FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_fname,
                                                    FINE_TUNING_MAXIMUM_FUSION_AVG_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'flatten':
                            weights_path = get_file(FINE_TUNING_MAXIMUM_FUSION_FLATTEN_fname,
                                                    FINE_TUNING_MAXIMUM_FUSION_FLATTEN_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

                        elif pooling_mode == 'max':
                            weights_path = get_file(FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_fname,
                                                    FINE_TUNING_MAXIMUM_FUSION_MAX_POOL_WEIGHTS_PATH,
                                                    cache_subdir=cache_subdir)

        else:
            if fusion_strategy == 'average':
                weights_path = get_file(FINE_TUNING_AVERAGE_FUSION_NO_TOP_fname,
                                        FINE_TUNING_AVERAGE_FUSION_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir=cache_subdir)

            elif fusion_strategy == 'concatenate':
                weights_path = get_file(FINE_TUNING_CONCATENATE_FUSION_NO_TOP_fname,
                                        FINE_TUNING_CONCATENATE_FUSION_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir=cache_subdir)

            elif fusion_strategy == 'maximum':
                weights_path = get_file(FINE_TUNING_MAXIMUM_FUSION_NO_TOP_fname,
                                        FINE_TUNING_MAXIMUM_FUSION_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir=cache_subdir)


        model.load_weights(weights_path)

    return model


