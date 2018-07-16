import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16,VGG19,ResNet50
from applications.vgg16_places_365 import VGG16_Places365
from keras.layers import Input


class FeatureExtraction():

    def __init__(self,
                 pre_trained_model):
        """
        Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples.
        These features are then run through a new classifier, which is trained from scratch.

        :param pre_trained_model: one of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`
        """

        # Base directory of raw jpg/png images
        base_dir = '/home/gkallia/git/Human-Rights-Archive-CNNs/datasets/Human_Rights_Archive_DB'

        train_dir = os.path.join(base_dir, 'train_val')
        test_dir = os.path.join(base_dir, 'test')
        self.nb_train_samples = 3050
        self.nb_test_samples = 270

        human_rights_classes = ['arms', 'child_labour', 'child_marriage', 'detention_centres',
                                'disability_rights', 'displaced_populations', 'environment',
                                'no_violation', 'out_of_school']

        # Augmentation configuration with only rescaling.
        # Rescale is a value by which we will multiply the data before any other processing.
        # Our original images consist in RGB coefficients in the 0-255, but such values would
        # be too high for our models to process (given a typical learning rate),
        # so we target values between 0 and 1 instead by scaling with a 1/255. factor.
        datagen = ImageDataGenerator(rescale=1. / 255)

        img_width, img_height = 224, 224

        self.batch_size = 25

        self.train_generator = datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                           classes=human_rights_classes, class_mode='categorical',
                                                           batch_size=self.batch_size)

        self.test_generator = datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                          classes=human_rights_classes, class_mode='categorical',
                                                          batch_size=self.batch_size)

        if not (pre_trained_model in {'VGG16', 'VGG19', 'ResNet50', 'VGG16_Places365'}):
            raise ValueError('The `pre_trained_model` argument should be either '
                             '`VGG16`, `VGG19`, `ResNet50`, '
                             'or `VGG16_Places365`. Other models will be supported in future releases. ')

        input_tensor = Input(shape=(224, 224, 3))

        # create the base pre-trained model for warm-up
        if pre_trained_model == 'VGG16':
            self.conv_base = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

        elif pre_trained_model == 'VGG19':
            self.conv_base = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)

        elif pre_trained_model == 'ResNet50':
            self.conv_base = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

        elif pre_trained_model == 'VGG16_Places365':
            self.conv_base = VGG16_Places365(weights='places', include_top=False, input_tensor=input_tensor)

        self.bottleneck_features_train_filename = 'bottleneck_features_train_' + pre_trained_model + '.npy'
        self.bottleneck_features_test_filename = 'bottleneck_features_test_' + pre_trained_model + '.npy'



    def save_bottlebeck_features(self):

        bottleneck_features_train = self.conv_base.predict_generator(self.train_generator, self.nb_train_samples // self.batch_size)

        np.save(open(self.bottleneck_features_train_filename, 'w'),bottleneck_features_train)

        bottleneck_features_test = self.conv_base.predict_generator(self.test_generator, self.nb_test_samples // 15)
        np.save(open(self.bottleneck_features_test_filename, 'w'),bottleneck_features_test)



    def train_classifier(self):

        train_data = np.load(open(self.bottleneck_features_train_filename, 'rb'))
        train_labels = np.array([0] * ( self.nb_train_samples // 9) + [1] * ( self.nb_train_samples // 9))

        test_data = np.load(open(self.bottleneck_features_test_filename, 'rb'))
        test_labels = np.array([0] * (self.nb_test_samples // 9) + [1] * ( self.nb_test_samples // 9))



import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained_model", type = str,help = 'One of `VGG`, `VGG19`, `ResNet50` or `VGG16_Places365`')


    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # --------- Configure and pass a tensorflow session to Keras to restrict GPU memory fraction --------- #
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.50
    set_session(tf.Session(config=config))

    args = get_args()

    feature_extraction = FeatureExtraction(pre_trained_model = args.pre_trained_model)

    feature_extraction.save_bottlebeck_features()



