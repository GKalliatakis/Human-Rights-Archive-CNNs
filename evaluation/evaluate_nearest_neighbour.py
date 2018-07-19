import numpy as np

from engine.feature_extraction import FeatureExtraction
from sklearn.neighbors import KNeighborsClassifier


import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_trained_model", type = str,help = 'One of `VGG16`, `VGG19`, `ResNet50` or `VGG16_Places365`')


    args = parser.parse_args()
    return args



args = get_args()
feature_extraction = FeatureExtraction(pre_trained_model=args.pre_trained_model)

# Uncomment to extract features
# train_features, train_labels, test_features, test_labels = feature_extraction.extract_bottlebeck_features()

train_features, train_labels, test_features, test_labels = feature_extraction.load_bottlebeck_features()


print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

if args.pre_trained_model == 'ResNet50':
    train_features = np.reshape(train_features, (3050, 7 * 7 * 2048))
    test_features = np.reshape(test_features, (270, 7 * 7 * 2048))

else:
    train_features = np.reshape(train_features, (3050, 7 * 7 * 512))
    test_features = np.reshape(test_features, (270, 7 * 7 * 512))

print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

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
