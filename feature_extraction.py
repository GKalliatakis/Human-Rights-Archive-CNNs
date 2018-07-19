# -*- coding: utf-8 -*-
"""Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples.
These features are then run through a new classifier, which is trained from scratch.

Reference
http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

"""
from __future__ import print_function
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16,VGG19,ResNet50
from applications.vgg16_places_365 import VGG16_Places365
from keras.layers import Input
from keras.utils.data_utils import get_file


VGG16_BOTTLENECK_TRAIN_FEATURES_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_train_features_VGG16.npy'
VGG16_BOTTLENECK_TRAIN_LABELS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_train_labels_VGG16.npy'
VGG16_BOTTLENECK_TEST_FEATURES_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_test_features_VGG16.npy'
VGG16_BOTTLENECK_TEST_LABELS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_test_labels_VGG16.npy'

VGG19_BOTTLENECK_TRAIN_FEATURES_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_train_features_VGG19.npy'
VGG19_BOTTLENECK_TRAIN_LABELS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_train_labels_VGG19.npy'
VGG19_BOTTLENECK_TEST_FEATURES_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_test_features_VGG19.npy'
VGG19_BOTTLENECK_TEST_LABELS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_test_labels_VGG19.npy'

ResNet50_BOTTLENECK_TRAIN_FEATURES_PATH = ''
ResNet50_BOTTLENECK_TRAIN_LABELS_PATH = ''
ResNet50_BOTTLENECK_TEST_FEATURES_PATH = ''
ResNet50_BOTTLENECK_TEST_LABELS_PATH = ''

VGG16_Places365_BOTTLENECK_TRAIN_FEATURES_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_train_features_VGG16_Places365.npy'
VGG16_Places365_BOTTLENECK_TRAIN_LABELS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_train_labels_VGG16_Places365.npy'
VGG16_Places365_BOTTLENECK_TEST_FEATURES_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_test_features_VGG16_Places365.npy'
VGG16_Places365_BOTTLENECK_TEST_LABELS_PATH = 'https://github.com/GKalliatakis/crispy-enigma/releases/download/v0.3/bottleneck_test_labels_VGG16_Places365.npy'


class FeatureExtraction():

    def __init__(self,
                 pre_trained_model):
        """
        Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples.
        These features are then run through a new classifier, which is trained from scratch.

        :param pre_trained_model: one of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`
        """

        # Base directory of raw jpg/png images
        # base_dir = '/home/gkallia/git/Human-Rights-Archive-CNNs/datasets/Human_Rights_Archive_DB'
        base_dir = '/home/sandbox/Desktop/Untitled'

        train_dir = os.path.join(base_dir, 'train_val')
        test_dir = os.path.join(base_dir, 'test')
        self.nb_train_samples = 3050
        self.nb_test_samples = 270

        # human_rights_classes = ['arms', 'child_labour', 'child_marriage', 'detention_centres',
        #                         'disability_rights', 'displaced_populations', 'environment',
        #                         'no_violation', 'out_of_school']

        # Augmentation configuration with only rescaling.
        # Rescale is a value by which we will multiply the data before any other processing.
        # Our original images consist in RGB coefficients in the 0-255, but such values would
        # be too high for our models to process (given a typical learning rate),
        # so we target values between 0 and 1 instead by scaling with a 1/255. factor.
        datagen = ImageDataGenerator(rescale=1. / 255)

        img_width, img_height = 224, 224

        self.train_batch_size = 25
        self.test_batch_size = 15
        self.train_generator = datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                           class_mode='categorical',
                                                           batch_size=self.train_batch_size)

        self.test_generator = datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                          class_mode='categorical',
                                                          batch_size=self.test_batch_size)

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

        self.bottleneck_train_features_filename = 'bottleneck_train_features_' + pre_trained_model + '.npy'
        self.bottleneck_train_labels_filename = 'bottleneck_train_labels_' + pre_trained_model + '.npy'
        self.bottleneck_test_features_filename = 'bottleneck_test_features_' + pre_trained_model + '.npy'
        self.bottleneck_test_labels_filename = 'bottleneck_test_labels_' + pre_trained_model + '.npy'

        self.cache_subdir = 'HRA_models'
        self.pre_trained_model = pre_trained_model



    def extract_bottlebeck_features(self):
        """Extracts bottleneck features for train and test sets.

            # Returns
                bottleneck_train_features : array-like, shape = (n_samples, n_features)
                    Train samples.
                train_labels : array-like, shape = (n_samples, n_outputs)
                    True labels for train samples.
                bottleneck_test_features : array-like, shape = (n_samples, n_features)
                    Test samples.
                test_labels : array-like, shape = (n_samples, n_outputs)
                    True labels for test samples.

        Reference
        http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

        """

        if self.pre_trained_model == 'ResNet50':
            bottleneck_train_features = np.zeros(shape=(self.nb_train_samples, 7, 7, 2048))
            bottleneck_test_features = np.zeros(shape=(self.nb_test_samples, 7, 7, 2048))

        else:
            bottleneck_train_features = np.zeros(shape=(self.nb_train_samples, 7, 7, 512))
            bottleneck_test_features = np.zeros(shape=(self.nb_test_samples, 7, 7, 512))

        train_labels = np.zeros(shape=(self.nb_train_samples, self.train_generator.num_classes))
        test_labels = np.zeros(shape=(self.nb_test_samples, self.test_generator.num_classes))

        i = 0
        for inputs_batch, labels_batch in self.train_generator:
            features_batch = self.conv_base.predict(inputs_batch)
            bottleneck_train_features[i * self.train_batch_size: (i + 1) * self.train_batch_size] = features_batch
            train_labels[i * self.train_batch_size: (i + 1) * self.train_batch_size] = labels_batch
            i += 1
            if i * self.train_batch_size >= self.nb_train_samples:
                # Note that since generators yield data indefinitely in a loop,
                # we must `break` after every image has been seen once.
                break

        np.save(open(self.bottleneck_train_features_filename, 'w'), bottleneck_train_features)
        np.save(open(self.bottleneck_train_labels_filename, 'w'), train_labels)


        j = 0
        for test_inputs_batch, test_labels_batch in self.test_generator:
            test_features_batch = self.conv_base.predict(test_inputs_batch)
            bottleneck_test_features[j * self.test_batch_size: (j + 1) * self.test_batch_size] = test_features_batch
            test_labels[j * self.test_batch_size: (j + 1) * self.test_batch_size] = test_labels_batch
            j += 1
            if j * self.test_batch_size >= self.nb_test_samples:
                # Note that since generators yield data indefinitely in a loop,
                # we must `break` after every image has been seen once.
                break

        np.save(open(self.bottleneck_test_features_filename, 'w'), bottleneck_test_features)
        np.save(open(self.bottleneck_test_labels_filename, 'w'), test_labels)


        return bottleneck_train_features, train_labels, bottleneck_test_features, test_labels



    def load_bottlebeck_features(self):
        """Loads previously saved bottleneck features for train and test sets.

            # Returns
                bottleneck_train_features : array-like, shape = (n_samples, n_features)
                    Train samples.
                train_labels : array-like, shape = (n_samples, n_outputs)
                    True labels for train samples.
                bottleneck_test_features : array-like, shape = (n_samples, n_features)
                    Test samples.
                test_labels : array-like, shape = (n_samples, n_outputs)
                    True labels for test samples.
        """


        # create the base pre-trained model for warm-up
        if self.pre_trained_model == 'VGG16':
            bottleneck_train_features_path = get_file('bottleneck_train_features_VGG16',
                                                      VGG16_BOTTLENECK_TRAIN_FEATURES_PATH,
                                                      cache_subdir=self.cache_subdir)

            bottleneck_train_labels_path = get_file('bottleneck_train_labels_VGG16',
                                                    VGG16_BOTTLENECK_TRAIN_LABELS_PATH,
                                                    cache_subdir=self.cache_subdir)

            bottleneck_test_features_path = get_file('bottleneck_test_features_VGG16',
                                                     VGG16_BOTTLENECK_TEST_FEATURES_PATH,
                                                     cache_subdir=self.cache_subdir)

            bottleneck_test_labels_path = get_file('bottleneck_test_labels_VGG16',
                                                   VGG16_BOTTLENECK_TEST_LABELS_PATH,
                                                   cache_subdir=self.cache_subdir)

        elif self.pre_trained_model == 'VGG19':
            bottleneck_train_features_path = get_file('bottleneck_train_features_VGG19',
                                                      VGG19_BOTTLENECK_TRAIN_FEATURES_PATH,
                                                      cache_subdir=self.cache_subdir)

            bottleneck_train_labels_path = get_file('bottleneck_train_labels_VGG19',
                                                    VGG19_BOTTLENECK_TRAIN_LABELS_PATH,
                                                    cache_subdir=self.cache_subdir)

            bottleneck_test_features_path = get_file('bottleneck_test_features_VGG19',
                                                     VGG19_BOTTLENECK_TEST_FEATURES_PATH,
                                                     cache_subdir=self.cache_subdir)

            bottleneck_test_labels_path = get_file('bottleneck_test_labels_VGG19',
                                                   VGG19_BOTTLENECK_TEST_LABELS_PATH,
                                                   cache_subdir=self.cache_subdir)


        elif self.pre_trained_model == 'ResNet50':
            bottleneck_train_features_path = get_file('bottleneck_train_features_ResNet50',
                                                      ResNet50_BOTTLENECK_TRAIN_FEATURES_PATH,
                                                      cache_subdir=self.cache_subdir)

            bottleneck_train_labels_path = get_file('bottleneck_train_labels_ResNet50',
                                                    ResNet50_BOTTLENECK_TRAIN_LABELS_PATH,
                                                    cache_subdir=self.cache_subdir)

            bottleneck_test_features_path = get_file('bottleneck_test_features_ResNet50',
                                                     ResNet50_BOTTLENECK_TEST_FEATURES_PATH,
                                                     cache_subdir=self.cache_subdir)

            bottleneck_test_labels_path = get_file('bottleneck_test_labels_ResNet50',
                                                   ResNet50_BOTTLENECK_TEST_LABELS_PATH,
                                                   cache_subdir=self.cache_subdir)


        elif self.pre_trained_model == 'VGG16_Places365':
            bottleneck_train_features_path = get_file('bottleneck_train_features_VGG16_Places365',
                                                      VGG16_Places365_BOTTLENECK_TRAIN_FEATURES_PATH,
                                                      cache_subdir=self.cache_subdir)

            bottleneck_train_labels_path = get_file('bottleneck_train_labels_VGG16_Places365',
                                                    VGG16_Places365_BOTTLENECK_TRAIN_LABELS_PATH,
                                                    cache_subdir=self.cache_subdir)

            bottleneck_test_features_path = get_file('bottleneck_test_features_VGG16_Places365',
                                                     VGG16_Places365_BOTTLENECK_TEST_FEATURES_PATH,
                                                     cache_subdir=self.cache_subdir)

            bottleneck_test_labels_path = get_file('bottleneck_test_labels_VGG16_Places365',
                                                   VGG16_Places365_BOTTLENECK_TEST_LABELS_PATH,
                                                   cache_subdir=self.cache_subdir)


        train_data = np.load(open(bottleneck_train_features_path, 'rb'))
        train_labels = np.load(open(bottleneck_train_labels_path, 'rb'))

        test_data = np.load(open(bottleneck_test_features_path, 'rb'))
        test_labels = np.load(open(bottleneck_test_labels_path, 'rb'))

        return train_data, train_labels, test_data, test_labels



import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained_model", type = str,help = 'One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # visualize confusion matrix on heat map
    # import pandas as pd
    # import seaborn as sn
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix
    #
    #
    # def print_cmx(y_true, y_pred):
    #     labels = sorted(list(set(y_true)))
    #     cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    #
    #     df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    #
    #     plt.figure(figsize=(10, 7))
    #     sn.heatmap(df_cmx, annot=True)
    #     plt.show()



    # --------- Configure and pass a tensorflow session to Keras to restrict GPU memory fraction --------- #
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.50
    set_session(tf.Session(config=config))

    # args = get_args()
    # feature_extraction = FeatureExtraction(pre_trained_model=args.pre_trained_model)

    pre_trained_model = 'ResNet50'
    # feature_extraction = FeatureExtraction(pre_trained_model=pre_trained_model)


    # train_features, train_labels, test_features, test_labels = feature_extraction.extract_bottlebeck_features()


    # train_features, train_labels, test_features, test_labels = feature_extraction.load_bottlebeck_features()

    train_features = np.load(open('/home/sandbox/Desktop/bottleneck_train_features_ResNet50.npy', 'rb'))
    train_labels = np.load(open('/home/sandbox/Desktop/bottleneck_train_labels_ResNet50.npy', 'rb'))

    test_features = np.load(open('/home/sandbox/Desktop/bottleneck_test_features_ResNet50.npy', 'rb'))
    test_labels = np.load(open('/home/sandbox/Desktop/bottleneck_test_labels_ResNet50.npy', 'rb'))

    print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)


    if pre_trained_model == 'ResNet50':
        train_features = np.reshape(train_features, (3050, 7 * 7 * 2048))
        test_features = np.reshape(test_features, (270, 7 * 7 * 2048))

    else:
        train_features = np.reshape(train_features, (3050, 7 * 7 * 512))
        test_features = np.reshape(test_features, (270, 7 * 7 * 512))

    print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)




    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV


    # Create and fit a nearest-neighbor classifier
    knn = KNeighborsClassifier()
    knn.fit(train_features, train_labels)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                         weights='uniform')

    # reference
    # https: // gurus.pyimagesearch.com / lesson - sample - k - nearest - neighbor - classification /

    # evaluate the model and update the accuracies list
    score = knn.score(test_features, test_labels)


    print ('KNeighborsClassifier mean accuracy:', score)

    from sklearn.svm import SVC

    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)

    svm_score = clf.score(test_features, test_labels)

    print('SVM mean accuracy:', svm_score)

