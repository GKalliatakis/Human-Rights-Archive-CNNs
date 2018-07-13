# -*- coding: utf-8 -*-
"""Trains a simple (baseline) deep CNN on the Human Rights Archive image dataset.
The baseline model is a simple stack of 3 convolution layers with a ReLU activation
and followed by max-pooling layers. This is very similar to the architectures that
Yann LeCun advocated in the 1990s for image classification (with the exception of ReLU).

# References
- [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]


|            Model           | Top-1 Mean Accuracy | Trainable params. |
|:--------------------------:|:-------------------:|:-----------------:|
| baseline-convnet 40 epochs |        0.1073       |     3,240,553     |

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 32)      896
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 32)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 32)      9248
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 32)        0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 64)        18496
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 64)        0
_________________________________________________________________
flatten (Flatten)            (None, 50176)             0
_________________________________________________________________
fc1 (Dense)                  (None, 64)                3211328
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0
_________________________________________________________________
predictions (Dense)          (None, 9)                 585
=================================================================
Total params: 3,240,553
Trainable params: 3,240,553
Non-trainable params: 0
_________________________________________________________________

"""

import os
import datetime

from keras.utils.data_utils import get_file
import keras.callbacks
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import get_source_inputs
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import CSVLogger


WEIGHTS_PATH_10_EPOCHS = 'https://github.com/GKalliatakis/expert-enigma/releases/download/0.1/baseline_model_10_epochs_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_40_EPOCHS = 'https://github.com/GKalliatakis/expert-enigma/releases/download/0.1/baseline_model_40_epochs_weights_tf_dim_ordering_tf_kernels.h5'


now = datetime.datetime.now

# Data preparation

# Base directory for the server
# base_dir = '/home/gkallia/git/Learning_Image_Representations_for_Recognising_HRV/datasets/HRA_uniform'

# Base directory for the local machine
base_dir = '/home/sandbox/git/caffe/python/Learning_Image_Representations_for_Recognising_HRV/Human-Rights-Violations-keras/datasets/TwoClass_HRA'

# Base directory for saving the trained models
base_dir_trained_models = '/home/gkallia/git/Learning_Image_Representations_for_Recognising_HRV/Human-Rights-Violations-keras/trained_models'
transfer_learning_dir = os.path.join(base_dir_trained_models, 'transfer_learning/')
fine_tune_dir = os.path.join(base_dir_trained_models, 'fine_tune/')
logs_dir = os.path.join(base_dir_trained_models, 'logs/')

trainval_dir = os.path.join(base_dir, 'TrainVal')
nb_train_samples = 1218

test_dir = os.path.join(base_dir, 'Test')
nb_test_samples = 100

trainval_augm_dir = os.path.join(base_dir, 'TrainValAugm')
nb_augmented_train_samples = 11323

human_rights_classes = ['Arms', 'ChildLabour', 'ChildMarriage', 'DetentionCentres', 'Disability', 'Environment',
                        'NoViolation', 'OutOfSchool', 'Refugees']

datagen = ImageDataGenerator(rescale=1. / 255)

train_batches = datagen.flow_from_directory(trainval_dir, target_size=(224, 224),
                                            classes=human_rights_classes,class_mode='categorical',
                                            batch_size=14)


augmented_train_batches = datagen.flow_from_directory(trainval_augm_dir, target_size=(224, 224),
                                                      classes=human_rights_classes,class_mode='categorical',
                                                      batch_size=13)


test_batches = datagen.flow_from_directory(test_dir, target_size=(224, 224),
                                           classes=human_rights_classes,class_mode='categorical',
                                           batch_size=10)



def top_3_categorical_accuracy(y_true, y_pred):
    """A metric function that is used to judge the top-3 performance of our model.
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def baseline_model(include_top=True, weights= None,
                   input_tensor=None,
                   classes=9,
                   epochs=40,
                   augmented_samples=False):
    """Instantiates the simple deep CNN architecture.

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

    TL_model_log = logs_dir + 'TwoClass_HRA_baseline' + epochs + '_log.csv'
    TL_csv_logger = CSVLogger(TL_model_log, append=True, separator=',')


    # dimensions of our images.
    img_width, img_height = 224, 224

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

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
        # x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='baseline_convnet')

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_categorical_accuracy])

    model.summary()


    # load weights
    if weights == 'HRA' and epochs == 10:
        if include_top:
            weights_path = get_file('baseline_model_10_epochs_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_10_EPOCHS,
                                    cache_subdir='models')

        model.load_weights(weights_path)

    elif weights == 'HRA' and epochs == 40:
        if include_top:
            weights_path = get_file('baseline_model_40_epochs_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_40_EPOCHS,
                                    cache_subdir='models')

        model.load_weights(weights_path)

    # When no weights are loaded, the model needs to be trained from scratch
    else:
        if augmented_samples:

            print('Using augmented samples for training. This may take a while ! \n')

            t = now()

            # steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch
            # finished and starting the next epoch. It should typically be equal to the number of samples of your dataset
            # divided by the batch size.
            history = model.fit_generator(augmented_train_batches,
                                          steps_per_epoch=nb_augmented_train_samples // 13,
                                          epochs=epochs,
                                          callbacks= [TL_csv_logger])

            print('Training time: %s' % (now() - t))
            weights_name = 'baseline_model_aug_' + str(epochs) + '_epochs.h5'

            model.save_weights('complete_models/' + weights_name)

            print('Model weights for the baseline model using augmented samples were saved as `' + weights_name + '`')

            test_loss_fully_fine_tuned, test_acc_fully_fine_tuned, top_3_categorical_accuracy_fully_fine_tuned = model.evaluate_generator(
                test_batches, steps=10,
                pickle_safe=True, workers=1)

            print('[AUGMENTED DATA] --> Top-1 test accuracy for baseline model: ', test_acc_fully_fine_tuned)
            print('[AUGMENTED DATA] --> Top-3 test accuracy for baseline model: ', top_3_categorical_accuracy_fully_fine_tuned)
            print('[AUGMENTED DATA] --> Test loss for fully for baseline model: ', test_loss_fully_fine_tuned)
            print ('\n')

        else:
            t = now()

            # steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch
            # finished and starting the next epoch. It should typically be equal to the number of samples of your dataset
            # divided by the batch size.
            history = model.fit_generator(train_batches,
                                          steps_per_epoch=nb_train_samples // 14,
                                          epochs=epochs,
                                          callbacks= [TL_csv_logger])



            print('Training time: %s' % (now() - t))
            weights_name = 'baseline_model_' + str(epochs) + '_epochs.h5'

            model.save_weights('complete_models/' + weights_name)

            print('Model weights for the baseline model were saved as `' + weights_name + '`')

            test_loss_fully_fine_tuned, test_acc_fully_fine_tuned, top_3_categorical_accuracy_fully_fine_tuned = model.evaluate_generator(
                test_batches, steps=10,
                pickle_safe=True, workers=1)

            print('Top-1 test accuracy for baseline model: ', test_acc_fully_fine_tuned)
            print('Top-3 test accuracy for baseline model: ', top_3_categorical_accuracy_fully_fine_tuned)
            print('Test loss for fully for baseline model: ', test_loss_fully_fine_tuned)
            print ('\n')

    return model


if __name__=="__main__":

    model = baseline_model(classes=9, epochs=10)
    model.summary()


