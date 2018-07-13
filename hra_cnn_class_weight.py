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

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import CSVLogger

import datetime
import keras.backend as K
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.engine.topology import get_source_inputs

# Preparation actions

now = datetime.datetime.now

# Base directory of raw jpg/png images
base_dir = '/home/gkallia/git/Human-Rights-Archive-CNNs/datasets/Human_Rights_Archive_DB'

# Base directory for saving the trained models
base_dir_trained_models = '/home/gkallia/git/Human-Rights-Archive-CNNs/trained_models'
baseline_dir = os.path.join(base_dir_trained_models, 'baseline/')
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



def baseline_model(include_top=True, weights=None,
                   input_tensor=None,
                   classes=9,
                   epochs=40,
                   data_augm_enabled=False):
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

    if weights == 'HRA' and include_top and classes != 9:
        raise ValueError('If using `weights` as Human Rights Archive, `classes` should be 9.')

    # Define the name of the model and its weights
    weights_name = 'cost_sensitive_baseline_' + str(epochs) + '_weights_tf_dim_ordering_tf_kernels.h5'

    augm_samples_weights_name = 'cost_sensitive_augm_baseline_' + str(epochs) + '_weights_tf_dim_ordering_tf_kernels.h5'

    model_log = logs_dir + 'cost_sensitive_baseline_' + str(epochs) + '_log.csv'
    csv_logger = CSVLogger(model_log, append=True, separator=',')

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
    model = Model(inputs, x, name='cost_sensitive_baseline_convnet')

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    if data_augm_enabled:

        print('Using augmented samples for training. This may take a while ! \n')

        t = now()

        # steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch
        # finished and starting the next epoch. It should typically be equal to the number of samples of your dataset
        # divided by the batch size.
        history = model.fit_generator(augmented_train_generator,
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=epochs,
                                      callbacks=[csv_logger],
                                      class_weight=class_weight)

        print('Training time: %s' % (now() - t))

        model.save_weights(baseline_dir + augm_samples_weights_name)

        print('Model weights for the baseline model using augmented samples were saved as `' + augm_samples_weights_name + '`')


    else:
        t = now()

        # steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch
        # finished and starting the next epoch. It should typically be equal to the number of samples of your dataset
        # divided by the batch size.
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=epochs,
                                      callbacks=[csv_logger],
                                      class_weight=class_weight)

        print('Training time: %s' % (now() - t))

        model.save_weights(baseline_dir + weights_name)

    return model




if __name__ == "__main__":

    baseline_model(epochs=10)

    baseline_model(epochs=20)

    baseline_model(epochs=40)


