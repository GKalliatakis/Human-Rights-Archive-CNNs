# -*- coding: utf-8 -*-
"""Transfer learning toy example.
"""
from applications.latest.hra_vgg16 import HRA_VGG16
from applications.latest.hra_vgg19 import HRA_VGG19
from applications.latest.hra_vgg16_places365 import HRA_VGG16_Places365
from applications.latest.compoundNet_vgg16 import CompoundNet_VGG16
from keras.applications.vgg16 import VGG16
from applications.vgg16_places_365 import VGG16_Places365

from keras.applications.vgg16 import VGG16
from applications.vgg16_places_365 import VGG16_Places365

from visualisations.grad_cam import Grad_CAM



# model = VGG16_Places365(weights='places', include_top=True)
model = VGG16(weights='imagenet')


# model = CompoundNet_VGG16(weights='HRA', mode='fine_tuning', pooling_mode='flatten', include_top=True,data_augm_enabled=False)
# # model= CompoundNet_VGG16(weights='HRA', mode= 'fine_tuning', fusion_strategy='average',  pooling_mode='avg', data_augm_enabled=False)
#
#
# model.save('early_fusion.h5')

# model = VGG16_Places365(weights='places')
# model.summary()
#
# #
# # ## Heatmaps of "class activation" example
# Grad_CAM('/home/sandbox/Desktop/Testing Images/child_labour_0008.jpg',
#          model,
#          conv_layer_to_visualise= 'block5_conv3',
#          to_file='child_labour_0008_CompoundNet_VGG16_fine_tuning_flatten_block5_conv3.png')


Grad_CAM('/media/sandbox/UBUNTU 16_0/100.jpg',
         model,
         conv_layer_to_visualise= 'block5_conv3',
         to_file='/media/sandbox/UBUNTU 16_0/superpixels_100.png')