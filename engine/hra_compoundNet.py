# -*- coding: utf-8 -*-
""" Combines image representations learned from object-centric and scene-centric
ConvNets in order to build an image recognition system and classify human rights violations.

Three different fusion strategies:
[A] Concatenation
[B] Averaging (aka sum pooling)
[C] Maximum pooling

"""

import os
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.callbacks import CSVLogger
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dropout
from keras.layers.merge import concatenate,average, maximum

from applications.vgg16_places_365 import VGG16_Places365

import datetime

# Preparation actions

now = datetime.datetime.now

# Base directory of raw jpg images
base_dir = '/home/gkallia/git/Learning_Image_Representations_for_Recognising_HRV/datasets/Human_Rights_Archive_DB'

# Base directory for saving the trained models
base_dir_trained_models = '/home/gkallia/git/Human-Rights-Violations-keras/trained_models/updated_lower_cost'
feature_extraction_dir = os.path.join(base_dir_trained_models, 'feature_extraction/')
fine_tuning_dir = os.path.join(base_dir_trained_models, 'fine_tuning/')
logs_dir = os.path.join(base_dir_trained_models, 'logs/')

train_dir = os.path.join(base_dir, 'train_val')
nb_train_samples = 3050

human_rights_classes = ['arms', 'child_labour', 'child_marriage', 'detention_centres',
                        'disability_rights', 'displaced_populations','environment',
                        'no_violation', 'out_of_school']

# https://groups.google.com/forum/#!topic/keras-users/MUO6v3kRHUw
# To train unbalanced classes 'fairly', we want to increase the importance of the under-represented class(es).
# To do this, we need to chose a reference class. You can pick any class to serve as the reference, but conceptually,
# I like the majority class (the one with the most samples).
# Creating your class_weight dictionary:
# 1. determine the ratio of reference_class/other_class. If you choose class_0 as your reference,
# you'll have (1000/1000, 1000/500, 1000/100) = (1,2,10)
# 2. map the class label to the ratio: class_weight={0:1, 1:2, 2:10}
class_weight = {0: 5.08, 1: 1, 2: 10.86, 3: 5.08, 4: 3.46, 5: 2.31, 6: 4.70, 7: 6.17, 8: 1.55}

# Augmentation configuration with only rescaling.
# Rescale is a value by which we will multiply the data before any other processing.
# Our original images consist in RGB coefficients in the 0-255, but such values would
# be too high for our models to process (given a typical learning rate),
# so we target values between 0 and 1 instead by scaling with a 1/255. factor.
datagen = ImageDataGenerator(rescale=1. / 255)

# This is the augmentation configuration we will use for training when augmented_samples is True
augmented_datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

img_width, img_height = 224, 224

batch_size = 25
feature_extraction_epochs = 10
fine_tune_epochs = 20

train_generator = datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                              classes=human_rights_classes, class_mode='categorical',
                                              batch_size=batch_size)

augmented_train_generator = augmented_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                                  classes=human_rights_classes, class_mode='categorical',
                                                                  batch_size=batch_size)



def compoundNet_feature_extraction(object_centric_model = 'VGG16',
                                   scene_centric_model = 'VGG16_Places365',
                                   fusion_strategy='concatenate',
                                   pooling_mode='avg',
                                   classes=9,
                                   data_augm_enabled = False):
    """ConvNet as fixed feature extractor, consist of taking the convolutional base of a previously-trained network,
    running the new data through it, and training a new classifier on top of the output.
    (i.e. train only the randomly initialized top layers while freezing all convolutional layers of the original model).

    # Arguments
        object_centric_model: one of `VGG16`, `VGG19` or `ResNet50`
        scene_centric_model: `VGG16_Places365`
        fusion_strategy: one of `concatenate` (feature vectors of different sources are concatenated into one super-vector),
            `average` (the feature set is averaged) or `maximum` (selects the highest value from the corresponding features).
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
        ValueError: in case of invalid argument for `object_centric_model`, `pooling_mode`,
        `fusion_strategy` , `scene_centric_model` or invalid input shape.
    """

    if not (object_centric_model in {'VGG16', 'VGG19', 'ResNet50'}):
        raise ValueError('The `scene_centric_model` argument should be either '
                         '`VGG16`, `VGG19` or `ResNet50`. Other models will be supported in future releases. ')

    if not (pooling_mode in {'avg', 'max', 'flatten'}):
        raise ValueError('The `pooling_mode` argument should be either '
                         '`avg` (GlobalAveragePooling2D), `max` '
                         '(GlobalMaxPooling2D), '
                         'or `flatten` (Flatten).')

    if not (fusion_strategy in {'concatenate', 'average', 'maximum'}):
        raise ValueError('The `fusion_strategy` argument should be either '
                         '`concatenate` (feature vectors of different sources are concatenated into one super-vector),'
                         ' `average` (the feature set is averaged) '
                         'or `maximum` (selects the highest value from the corresponding features).')

    if not (scene_centric_model in {'VGG16_Places365'}):
        raise ValueError('The `scene_centric_model` argument should be '
                         '`VGG16_Places365`. Other models will be supported in future releases.')

    # Define the name of the model and its weights
    weights_name = 'compoundNet_feature_extraction_' \
                   + object_centric_model + '_' \
                   + fusion_strategy + '_fusion_' \
                   + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'


    augm_samples_weights_name = 'augm_compoundNet_feature_extraction_' \
                                + object_centric_model + '_' \
                                + fusion_strategy + '_fusion_' \
                                + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'

    model_log = logs_dir + 'compoundNet_feature_extraction_' \
                                + object_centric_model + '_' \
                                + fusion_strategy + '_fusion_' \
                                + pooling_mode + '_pool_log.csv'
    csv_logger = CSVLogger(model_log, append=True, separator=',')

    input_tensor = Input(shape=(224, 224, 3))


    # create the base object_centric_model pre-trained model for warm-up
    if object_centric_model == 'VGG16':
        object_base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)


    elif object_centric_model == 'VGG19':
        object_base_model = VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)


    elif object_centric_model == 'ResNet50':
        tmp_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        object_base_model = Model(inputs=tmp_model.input, outputs=tmp_model.get_layer('activation_48').output)


    print ('\n \n')
    print('The plain, object-centric `' + object_centric_model + '` pre-trained convnet was successfully initialised.\n')

    scene_base_model = VGG16_Places365(input_tensor=input_tensor,weights='places', include_top=False)

    print('The plain, scene-centric `' + scene_centric_model + '` pre-trained convnet was successfully initialised.\n')

    # retrieve the ouputs
    object_base_model_output = object_base_model.output
    scene_base_model_output = scene_base_model.output

    # We will feed the extracted features to a merging layer
    if fusion_strategy == 'concatenate':
        merged = concatenate([object_base_model_output, scene_base_model_output])

    elif fusion_strategy == 'average':
        merged = average([object_base_model_output, scene_base_model_output])

    else:
        merged = maximum([object_base_model_output, scene_base_model_output])


    if pooling_mode == 'avg':
        x = GlobalAveragePooling2D(name='GAP')(merged)
    elif pooling_mode == 'max':
        x = GlobalMaxPooling2D(name='GMP')(merged)
    elif pooling_mode =='flatten':
        x = Flatten(name='FLATTEN')(merged)


    x = Dense(256, activation='relu', name='FC1')(x)  # let's add a fully-connected layer

    # When random init is enabled, we want to include Dropout,
    # otherwise when loading a pre-trained HRA model we want to omit
    # Dropout layer so the visualisations are done properly (there is an issue if it is included)
    x = Dropout(0.5,name='DROPOUT')(x)
    # and a logistic layer with the number of classes defined by the `classes` argument
    predictions = Dense(classes, activation='softmax', name='PREDICTIONS')(x)  # new softmax layer

    # this is the transfer learning model we will train
    model = Model(inputs=object_base_model.input, outputs=predictions)


    print('Randomly initialised classifier was successfully added on top of the merged outputs. \n')

    print('Number of trainable weights before freezing the conv. bases of the respective original models: '
          '' + str(len(model.trainable_weights)))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers of the preliminary base model
    for layer in object_base_model.layers:
        layer.trainable = False

    for layer in scene_base_model.layers:
        layer.trainable = False

    print('Number of trainable weights after freezing the conv. bases of the respective original models: '
          '' + str(len(model.trainable_weights)))

    print ('\n')

    # compile the warm_up_model (should be done *after* setting layers to non-trainable)

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # # The attribute model.metrics_names will give you the display labels for the scalar outputs.
    # print warm_up_model.metrics_names

    if data_augm_enabled:
        print('Using augmented samples for training. This may take a while ! \n')

        t = now()

        history = model.fit_generator(augmented_train_generator,
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=feature_extraction_epochs,
                                      callbacks=[csv_logger],
                                      class_weight=class_weight)

        print('Training time for re-training the last Dense layer using augmented samples: %s' % (now() - t))

        model.save_weights(feature_extraction_dir + augm_samples_weights_name)
        print(
            'Model weights using augmented samples were saved as `' + augm_samples_weights_name + '`')
        print ('\n')


    else:
        t = now()
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=feature_extraction_epochs,
                                      callbacks=[csv_logger],
                                      class_weight=class_weight)

        print('Training time for re-training the last Dense layer: %s' % (now() - t))

        model.save_weights(feature_extraction_dir + weights_name)
        print('Model weights were saved as `' + weights_name + '`')
        print ('\n')

    return model


def compoundNet_fine_tuning(stable_model,
                            object_centric_model='VGG16',
                            scene_centric_model='VGG16_Places365',
                            fusion_strategy='concatenate',
                            pooling_mode='avg',
                            data_augm_enabled=False):
    """`Fine-tuning`, consists in unfreezing a few of the top layers (in our case, unfreeze the 2 lower convolutional layers)
    of a frozen model base used for feature extraction,
    and jointly training both the newly added part of the model (in our case, the fully-connected classifier) and these top layers.

    # Arguments
        stable_model: a previously trained top-level classifier -- `transfer_learning` function needs to be utilised for that
        object_centric_model: one of `VGG16`, `VGG19` or `ResNet50`
        scene_centric_model: `VGG16_Places365`
        fusion_strategy: one of `concatenate` (feature vectors of different sources are concatenated into one super-vector),
            `average` (the feature set is averaged) or `maximum` (selects the highest value from the corresponding features).
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
                be applied..
        data_augm_enabled: whether to use the augmented samples during training

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `object_centric_model`, `pooling_mode`,
        `fusion_strategy` , `scene_centric_model` or invalid input shape.

    """

    if not (object_centric_model in {'VGG16', 'VGG19', 'ResNet50'}):
        raise ValueError('The `scene_centric_model` argument should be either '
                         '`VGG16`, `VGG19` or `ResNet50`. Other models will be supported in future releases. ')

    if not (pooling_mode in {'avg', 'max', 'flatten'}):
        raise ValueError('The `pooling_mode` argument should be either '
                         '`avg` (GlobalAveragePooling2D), `max` '
                         '(GlobalMaxPooling2D), '
                         'or `flatten` (Flatten).')

    if not (fusion_strategy in {'concatenate', 'average', 'maximum'}):
        raise ValueError('The `fusion_strategy` argument should be either '
                         '`concatenate` (feature vectors of different sources are concatenated into one super-vector),'
                         ' `average` (the feature set is averaged) '
                         'or `maximum` (selects the highest value from the corresponding features).')

    if not (scene_centric_model in {'VGG16_Places365'}):
        raise ValueError('The `scene_centric_model` argument should be '
                         '`VGG16_Places365`. Other models will be supported in future releases.')



    # Define the name of the model and its weights
    weights_name = 'compoundNet_fine_tuning_' \
                   + object_centric_model + '_' \
                   + fusion_strategy + '_fusion_' \
                   + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'

    augm_samples_weights_name = 'augm_compoundNet_fine_tuning_' \
                                + object_centric_model + '_' \
                                + fusion_strategy + '_fusion_' \
                                + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'

    model_log = logs_dir + 'compoundNet_fine_tuning_' \
                + object_centric_model + '_' \
                + fusion_strategy + '_fusion_' \
                + pooling_mode + '_pool_log.csv'


    csv_logger = CSVLogger(model_log, append=True, separator=',')

    # create the base pre-trained model for warm-up
    if object_centric_model == 'VGG16':
        nb_of_layers_to_freeze = 33

    elif object_centric_model == 'VGG19':
        nb_of_layers_to_freeze = 36

    elif object_centric_model == 'ResNet50':
        nb_of_layers_to_freeze = 184

    print('Stable CompoundNet model, with properly trained last Dense layer, was successfully loaded. \n')

    print('Number of trainable weights before unfreezing the last conv. block of the stable model: '
          '' + str(len(stable_model.trainable_weights)))

    # we will freeze the `first nb_of_layers_to_freeze` layers and unfreeze the rest:
    for layer in stable_model.layers[:nb_of_layers_to_freeze]:
        layer.trainable = False
    for layer in stable_model.layers[nb_of_layers_to_freeze:]:
        layer.trainable = True

    print('Number of trainable weights after unfreezing the last conv. block of the stable model: '
          '' + str(len(stable_model.trainable_weights)))

    # compile the model again (should be done *after* setting layers to non-trainable)
    stable_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    stable_model.summary()

    # # The attribute model.metrics_names will give you the display labels for the scalar outputs.
    # print warm_up_model.metrics_names

    if data_augm_enabled:
        print('Using augmented samples for training. This may take a while ! \n')

        t = now()

        history = stable_model.fit_generator(augmented_train_generator,
                                             steps_per_epoch=nb_train_samples // batch_size,
                                             epochs=fine_tune_epochs,
                                             callbacks=[csv_logger],
                                             class_weight=class_weight)

        print('Training time for fine-tuning using augmented samples: %s' % (now() - t))

        stable_model.save_weights(fine_tuning_dir + augm_samples_weights_name)
        print(
            'Model weights using augmented samples were saved as `' + augm_samples_weights_name + '`')
        print ('\n')


    else:
        t = now()
        history = stable_model.fit_generator(train_generator,
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=fine_tune_epochs,
                                      callbacks=[csv_logger],
                                      class_weight=class_weight)

        print('Training time for fine-tuning: %s' % (now() - t))

        stable_model.save_weights(fine_tuning_dir + weights_name)
        print('Model weights were saved as `' + weights_name + '`')
        print ('\n')


    return stable_model


if __name__ == "__main__":


    scene_centric_model = 'VGG16_Places365'

    object_centric_model = 'VGG19'


    transfer_learning_model = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='average',
                                                             pooling_mode='avg',
                                                             data_augm_enabled=True)


    fine_tuned_model = compoundNet_fine_tuning(transfer_learning_model,
                                               object_centric_model=object_centric_model,
                                               scene_centric_model=scene_centric_model,
                                               fusion_strategy='average',
                                               pooling_mode='avg',
                                               data_augm_enabled=True)




    transfer_learning_model2 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='average',
                                                             pooling_mode='flatten',
                                                             data_augm_enabled=True)


    fine_tuned_model2 = compoundNet_fine_tuning(transfer_learning_model2,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='average',
                                                pooling_mode='flatten',
                                                data_augm_enabled=True)



    transfer_learning_model3 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='average',
                                                             pooling_mode='max',
                                                             data_augm_enabled=True)

    fine_tuned_model3 = compoundNet_fine_tuning(transfer_learning_model3,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='average',
                                                pooling_mode='max',
                                                data_augm_enabled=True)

    transfer_learning_model4 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='concatenate',
                                                             pooling_mode='avg',
                                                             data_augm_enabled=True)


    fine_tuned_model4 = compoundNet_fine_tuning(transfer_learning_model4,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='concatenate',
                                                pooling_mode='avg',
                                                data_augm_enabled=True)




    transfer_learning_model5 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='concatenate',
                                                             pooling_mode='flatten',
                                                             data_augm_enabled=True)


    fine_tuned_model5 = compoundNet_fine_tuning(transfer_learning_model5,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='concatenate',
                                                pooling_mode='flatten',
                                                data_augm_enabled=True)



    transfer_learning_model6 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='concatenate',
                                                             pooling_mode='max',
                                                             data_augm_enabled=True)


    fine_tuned_model6 = compoundNet_fine_tuning(transfer_learning_model6,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='concatenate',
                                                pooling_mode='max',
                                                data_augm_enabled=True)

    transfer_learning_model7 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='maximum',
                                                             pooling_mode='avg',
                                                             data_augm_enabled=True)


    fine_tuned_model7 = compoundNet_fine_tuning(transfer_learning_model7,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='maximum',
                                                pooling_mode='avg',
                                                data_augm_enabled=True)




    transfer_learning_model8 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='maximum',
                                                             pooling_mode='flatten',
                                                             data_augm_enabled=True)


    fine_tuned_model8 = compoundNet_fine_tuning(transfer_learning_model8,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='maximum',
                                                pooling_mode='flatten',
                                                data_augm_enabled=True)



    transfer_learning_model9 = compoundNet_feature_extraction(object_centric_model = object_centric_model,
                                                             scene_centric_model = scene_centric_model,
                                                             fusion_strategy='maximum',
                                                             pooling_mode='max',
                                                             data_augm_enabled=True)


    fine_tuned_model9 = compoundNet_fine_tuning(transfer_learning_model9,
                                                object_centric_model=object_centric_model,
                                                scene_centric_model=scene_centric_model,
                                                fusion_strategy='maximum',
                                                pooling_mode='max',
                                                data_augm_enabled=True)