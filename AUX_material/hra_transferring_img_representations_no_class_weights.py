# -*- coding: utf-8 -*-
"""Leverage a pre-trained network (saved network previously trained on a large dataset)
in order to build an image recognition system and classify human rights violations.

Transfer image representations from popular deep learning models using the (original 9 classes) HRA dataset:

[A] ConvNet as fixed feature extractor.`Feature extraction` will simply consist of taking the convolutional base
of a previously-trained network, running the new data through it, and training a new classifier on top of the output.
(i.e. train only the randomly initialized top layers while freezing all convolutional layers of the original model).

[B] `Fine-tuning`, consists in unfreezing a few of the top layers (in our case, unfreeze the 2 lower convolutional layers)
of a frozen model base used for feature extraction,
and jointly training both the newly added part of the model (in our case, the fully-connected classifier) and these top layers.

Doing both, in that order, will ensure a more stable and consistent training.
This is because the large gradient updates triggered by randomly initialized weights could wreck
the learned weights in the convolutional base if not frozen.
Once the last layer has stabilized (transfer learning), then we move onto retraining more layers (fine-tuning).

[1] In order to perform fine-tuning, all layers should start with properly trained weights:
for instance you should not slap a randomly initialized fully-connected network on
top of a pre-trained convolutional base. This is because the large gradient updates triggered
by the randomly initialized weights would wreck the learned weights in the convolutional base.
In our case this is why we first train the top-level classifier, and only then start fine-tuning convolutional weights alongside it.

[2] We choose to only fine-tune the last convolutional block rather than the entire network
in order to prevent overfitting, since the entire network would have a very large entropic capacity and
thus a strong tendency to overfit. The features learned by low-level convolutional blocks are more general,
less abstract than those found higher-up, so it is sensible to keep the first
few blocks fixed (more general features) and only fine-tune the last one (more specialized features).

[3] Fine-tuning should be done with a very slow learning rate, and typically with the SGD optimizer
rather than an adaptative learning rate optimizer such as RMSProp. This is to make sure that the
magnitude of the updates stays very small, so as not to wreck the previously learned features.

# References
- [https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2]
- [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]

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

from applications.vgg16_places_365 import VGG16_Places365

import datetime

# Preparation actions

now = datetime.datetime.now

# Base directory of raw jpg/png images
base_dir = '/home/gkallia/git/Human-Rights-Archive-CNNs/datasets/Human_Rights_Archive_DB'

# Base directory for saving the trained models
base_dir_trained_models = '/home/gkallia/git/Human-Rights-Archive-CNNs/trained_models'
feature_extraction_dir = os.path.join(base_dir_trained_models, 'feature_extraction/')
fine_tuning_dir = os.path.join(base_dir_trained_models, 'fine_tuning/')
logs_dir = os.path.join(base_dir_trained_models, 'logs/')

train_dir = os.path.join(base_dir, 'train_val')
nb_train_samples = 3050

human_rights_classes = ['arms', 'child_labour', 'child_marriage', 'detention_centres',
                        'disability_rights', 'displaced_populations','environment',
                        'no_violation', 'out_of_school']

# Augmentation configuration with only rescaling.
# Rescale is a value by which we will multiply the data before any other processing.
# Our original images consist in RGB coefficients in the 0-255, but such values would
# be too high for our models to process (given a typical learning rate),
# so we target values between 0 and 1 instead by scaling with a 1/255. factor.
datagen = ImageDataGenerator(rescale=1. / 255)

# This is the augmentation configuration we will use for training when data_augm_enabled argument is True
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



def feature_extraction(pre_trained_model='VGG16',
                       pooling_mode='avg',
                       classes=9,
                       data_augm_enabled = False):
    """ConvNet as fixed feature extractor, consist of taking the convolutional base of a previously-trained network,
    running the new data through it, and training a new classifier on top of the output.
    (i.e. train only the randomly initialized top layers while freezing all convolutional layers of the original model).

    # Arguments
        pre_trained_model: one of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`
        pooling_mode: Optional pooling_mode mode for feature extraction
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
        data_augm_enabled: whether to augment the samples during training

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `pre_trained_model`, `pooling_mode` or invalid input shape.
    """


    if not (pre_trained_model in {'VGG16', 'VGG19', 'ResNet50', 'VGG16_Places365'}):
        raise ValueError('The `pre_trained_model` argument should be either '
                         '`VGG16`, `VGG19`, `ResNet50`, '
                         'or `VGG16_Places365`. Other models will be supported in future releases. ')

    if not (pooling_mode in {'avg', 'max', 'flatten'}):
        raise ValueError('The `pooling_mode` argument should be either '
                         '`avg` (GlobalAveragePooling2D), `max` '
                         '(GlobalMaxPooling2D), '
                         'or `flatten` (Flatten).')

    # Define the name of the model and its weights
    weights_name = 'feature_extraction_' + pre_trained_model + '_' + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'

    augm_samples_weights_name = 'augm_feature_extraction_' + pre_trained_model + '_' + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'

    model_log = logs_dir + 'feature_extraction_' + pre_trained_model + '_' + pooling_mode + '_pool_log.csv'
    csv_logger = CSVLogger(model_log, append=True, separator=',')

    input_tensor = Input(shape=(224, 224, 3))

    # create the base pre-trained model for warm-up
    if pre_trained_model == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    elif pre_trained_model == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)

    elif pre_trained_model == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    elif pre_trained_model == 'VGG16_Places365':
        base_model = VGG16_Places365(weights='places', include_top=False, input_tensor=input_tensor)

    print ('\n \n')
    print('The plain `' + pre_trained_model + '` pre-trained convnet was successfully initialised.\n')


    x = base_model.output

    # Now we set-up transfer learning process - freeze all but the penultimate layer
    # and re-train the last Dense layer with 9 final outputs representing probabilities for HRA classes.
    # Build a  randomly initialised classifier model to put on top of the convolutional model

    # both `avg`and `max`result in the same size of the Dense layer afterwards
    # Both Flatten and GlobalAveragePooling2D are valid options. So is GlobalMaxPooling2D.
    # Flatten will result in a larger Dense layer afterwards, which is more expensive
    # and may result in worse overfitting. But if you have lots of data, it might also perform better.
    # https://github.com/keras-team/keras/issues/8470
    if pooling_mode == 'avg':
        x = GlobalAveragePooling2D(name='GAP')(x)
    elif pooling_mode == 'max':
        x = GlobalMaxPooling2D(name='GMP')(x)
    elif pooling_mode == 'flatten':
        x = Flatten(name='FLATTEN')(x)


    x = Dense(256, activation='relu', name='FC1')(x)  # let's add a fully-connected layer

    # When random init is enabled, we want to include Dropout,
    # otherwise when loading a pre-trained HRA model we want to omit
    # Dropout layer so the visualisations are done properly (there is an issue if it is included)
    x = Dropout(0.5, name='DROPOUT')(x)
    # and a logistic layer with the number of classes defined by the `classes` argument
    predictions = Dense(classes, activation='softmax', name='PREDICTIONS')(x)  # new softmax layer

    # this is the transfer learning model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    print('Randomly initialised classifier was successfully added on top of the original pre-trained conv. base. \n')

    print('Number of trainable weights before freezing the conv. base of the original pre-trained convnet: '
          '' + str(len(model.trainable_weights)))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers of the preliminary base model
    for layer in base_model.layers:
        layer.trainable = False

    print('Number of trainable weights after freezing the conv. base of the pre-trained convnet: '
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
                                      callbacks=[csv_logger])

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
                                      callbacks=[csv_logger])

        print('Training time for re-training the last Dense layer: %s' % (now() - t))

        model.save_weights(feature_extraction_dir + weights_name)
        print('Model weights were saved as `' + weights_name + '`')
        print ('\n')

    return model


def fine_tuning(stable_model,
                pre_trained_model='VGG16',
                pooling_mode='avg',
                data_augm_enabled=False):
    """`Fine-tuning`, consists in unfreezing a few of the top layers (in our case, unfreeze the 2 lower convolutional layers)
    of a frozen model base used for feature extraction,
    and jointly training both the newly added part of the model (in our case, the fully-connected classifier) and these top layers.

    # Arguments
        stable_model: a previously trained top-level classifier -- `transfer_learning` function needs to be utilised for that
        pre_trained_model: one of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`
        pooling_mode: Optional pooling_mode mode for feature extraction
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling_mode
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling_mode will
                be applied.
        data_augm_enabled: whether to augment the samples during training

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `pre_trained_model`, `pooling_mode` or invalid input shape.

    """

    if not (pre_trained_model in {'VGG16', 'VGG19', 'ResNet50', 'VGG16_Places365'}):
        raise ValueError('The `pre_trained_model` argument should be either '
                         '`VGG16`, `VGG19`, `ResNet50`, '
                         'or `VGG16_Places365`. Other models will be supported in future releases. ')

    if not (pooling_mode in {'avg', 'max', 'flatten'}):
        raise ValueError('The `pooling_mode` argument should be either '
                         '`avg` (GlobalAveragePooling2D), `max` '
                         '(GlobalMaxPooling2D), '
                         'or `flatten` (Flatten).')

    # Define the name of the model and its weights
    weights_name = 'fine_tuning_' + pre_trained_model + '_' + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'

    augm_samples_weights_name = 'augm_fine_tuning_' + pre_trained_model + '_' + pooling_mode + '_pool_weights_tf_dim_ordering_tf_kernels.h5'


    model_log = logs_dir + 'fine_tuning_' + pre_trained_model + '_' + pooling_mode + '_pool_log.csv'
    csv_logger = CSVLogger(model_log, append=True, separator=',')

    # create the base pre-trained model for warm-up
    if pre_trained_model == 'VGG16':
        if pooling_mode== 'flatten':
            nb_of_layers_to_freeze = 17
        else:
            nb_of_layers_to_freeze = 16

    elif pre_trained_model == 'VGG19':
        if pooling_mode== 'flatten':
            nb_of_layers_to_freeze = 20
        else:
            nb_of_layers_to_freeze = 19

    elif pre_trained_model == 'ResNet50':
        nb_of_layers_to_freeze = 163

    elif pre_trained_model == 'VGG16_Places365':
        if pooling_mode== 'flatten':
            nb_of_layers_to_freeze = 17
        else:
            nb_of_layers_to_freeze = 16

    print('Stable model with properly trained last Dense layer was successfully loaded. \n')

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
                                             callbacks=[csv_logger])

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
                                      callbacks=[csv_logger])

        print('Training time for fine-tuning: %s' % (now() - t))

        stable_model.save_weights(fine_tuning_dir + weights_name)
        print('Model weights were saved as `' + weights_name + '`')
        print ('\n')


    return stable_model


if __name__ == "__main__":
    pre_trained_model = sys.argv[1]


    transfer_learning_model = feature_extraction(pre_trained_model=pre_trained_model,
                                                 pooling_mode='avg',
                                                 data_augm_enabled=True)


    fine_tuned_model = fine_tuning(transfer_learning_model,
                                   pre_trained_model=pre_trained_model,
                                   pooling_mode='avg',
                                   data_augm_enabled=True)




    transfer_learning_model2 = feature_extraction(pre_trained_model=pre_trained_model,
                                                  pooling_mode='flatten',
                                                  data_augm_enabled=True)


    fine_tuned_model2 = fine_tuning(transfer_learning_model2,
                                    pre_trained_model=pre_trained_model,
                                    pooling_mode='flatten',
                                    data_augm_enabled=True)




    transfer_learning_model3 = feature_extraction(pre_trained_model=pre_trained_model,
                                                  pooling_mode='max',
                                                  data_augm_enabled=True)


    fine_tuned_model3 = fine_tuning(transfer_learning_model3,
                                    pre_trained_model=pre_trained_model,
                                    pooling_mode='max',
                                    data_augm_enabled=True)



