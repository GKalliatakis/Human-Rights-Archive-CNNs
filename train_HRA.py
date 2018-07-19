"""Abstract script for two-phase training of various CNNs on the HRA dataset.


    Example
    --------
    >>> python train_HRA.py --pre_trained_model VGG16 --pooling_mode avg

# Reference:
- [Exploring object-centric and scene-centric CNN features and their complementarity for human rights violations recognition in images](https://arxiv.org/pdf/1805.04714.pdf)

"""

from __future__ import print_function

import argparse

from AUX_material.hra_transferring_img_representations_no_class_weights import feature_extraction as fe
from AUX_material.hra_transferring_img_representations_no_class_weights import fine_tuning as ft
from engine.hra_transferring_img_representations import feature_extraction as fe_class_weights
from engine.hra_transferring_img_representations import fine_tuning as ft_class_weights


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained_model", type = str,help = 'One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`')
    parser.add_argument("--pooling_mode", type = str, help = 'One of `avg`, `max`, or `flatten`')
    parser.add_argument("--data_augm_enabled", type = bool, default = False, help = 'Whether to augment the samples during training or not')
    parser.add_argument("--include_class_weight", type=bool, default=True,
                        help='Dictionary mapping class indices (integers) to a weight (float) value, '
                             'used for weighting the loss function (during training only)')

    args = parser.parse_args()
    return args


# --------- Configure and pass a tensorflow session to Keras to restrict GPU memory fraction --------- #
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))


args = get_args()


if args.include_class_weight == False:

    first_phase_model = fe(pre_trained_model=args.pre_trained_model,
                           pooling_mode=args.pooling_mode,
                           data_augm_enabled=args.data_augm_enabled)

    second_phase_model = ft(stable_model=first_phase_model,
                            pre_trained_model=args.pre_trained_model,
                            pooling_mode=args.pooling_mode,
                            data_augm_enabled=args.data_augm_enabled)

else:

    first_phase_model = fe_class_weights(pre_trained_model=args.pre_trained_model,
                                         pooling_mode=args.pooling_mode,
                                         data_augm_enabled=args.data_augm_enabled)

    second_phase_model = ft_class_weights(stable_model=first_phase_model,
                                          pre_trained_model=args.pre_trained_model,
                                          pooling_mode=args.pooling_mode,
                                          data_augm_enabled=args.data_augm_enabled)
