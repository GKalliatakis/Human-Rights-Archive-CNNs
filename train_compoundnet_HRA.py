"""Abstract script for two-phase training of CompoundNet on the HRA dataset.


    Example
    --------
    >>> python train_compoundnet_HRA.py --object_centric_model VGG16 --fusion_strategy average --pooling_mode avg

# Reference:
- [Exploring object-centric and scene-centric CNN features and their complementarity for human rights violations recognition in images](https://arxiv.org/pdf/1805.04714.pdf)

"""

from __future__ import print_function
from engine.hra_compoundNet import compoundNet_feature_extraction, compoundNet_fine_tuning


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_centric_model", type = str,help = 'One of `VGG16`, `VGG19` or `ResNet50`')
    parser.add_argument("--scene_centric_model", type=str, default = 'VGG16_Places365', help='Only `VGG16_Places365` at the moment')
    parser.add_argument("--fusion_strategy", type=str, help='one of `concatenate`, `average` or `maximum` ')
    parser.add_argument("--pooling_mode", type = str, help = 'One of `avg`, `max`, or `flatten`')
    parser.add_argument("--data_augm_enabled", type = bool, default = False, help = 'Whether to augment the samples during training or not')


    args = parser.parse_args()
    return args


# --------- Configure and pass a tensorflow session to Keras to restrict GPU memory fraction --------- #
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))


args = get_args()

first_phase_model = compoundNet_feature_extraction(object_centric_model=args.object_centric_model,
                                                   scene_centric_model=args.scene_centric_model,
                                                   fusion_strategy=args.fusion_strategy,
                                                   pooling_mode=args.pooling_mode,
                                                   data_augm_enabled=args.data_augm_enabled)

second_phase_model = compoundNet_fine_tuning(first_phase_model,
                                             object_centric_model=args.object_centric_model,
                                             scene_centric_model=args.scene_centric_model,
                                             fusion_strategy=args.fusion_strategy,
                                             pooling_mode=args.pooling_mode,
                                             data_augm_enabled=args.data_augm_enabled)
