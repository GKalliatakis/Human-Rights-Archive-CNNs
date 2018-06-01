"""Custom metrics.
"""

import numpy as np
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy



def top_3_categorical_accuracy(y_true, y_pred):
    """A metric function that is used to judge the top-3 performance of our model.
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)



def single_class_accuracy(interesting_class_id):
    # https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn


#------------------------------------------------------------------------------------------------#

# from sklearn.utils import check_array, check_consistent_length
# from sklearn.utils.multiclass import type_of_target
# def coverage_error(y_true, y_pred, sample_weight=None):
#     """Coverage error measure
#     Compute how far we need to go through the ranked scores to cover all
#     true labels. The best value is equal to the average number
#     of labels in ``y_true`` per sample.
#     Ties in ``y_scores`` are broken by giving maximal rank that would have
#     been assigned to all tied values.
#     Note: Our implementation's score is 1 greater than the one given in
#     Tsoumakas et al., 2010. This extends it to handle the degenerate case
#     in which an instance has 0 true labels.
#     Read more in the :ref:`User Guide <coverage_error>`.
#     Parameters
#     ----------
#     y_true : array, shape = [n_samples, n_labels]
#         True binary labels in binary indicator format.
#     y_score : array, shape = [n_samples, n_labels]
#         Target scores, can either be probability estimates of the positive
#         class, confidence values, or non-thresholded measure of decisions
#         (as returned by "decision_function" on some classifiers).
#     sample_weight : array-like of shape = [n_samples], optional
#         Sample weights.
#     Returns
#     -------
#     coverage_error : float
#     References
#     ----------
#     .. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
#            Mining multi-label data. In Data mining and knowledge discovery
#            handbook (pp. 667-685). Springer US.
#     """
#
#
#     y_true = check_array(y_true, ensure_2d=False)
#     y_score = check_array(y_pred, ensure_2d=False)
#     check_consistent_length(y_true, y_score, sample_weight)
#
#     y_type = type_of_target(y_true)
#     if y_type != "multilabel-indicator":
#         raise ValueError("{0} format is not supported".format(y_type))
#
#     if y_true.shape != y_score.shape:
#         raise ValueError("y_true and y_score have different shape")
#
#     y_score_mask = np.ma.masked_array(y_score, mask=np.logical_not(y_true))
#     y_min_relevant = y_score_mask.min(axis=1).reshape((-1, 1))
#     coverage = (y_score >= y_min_relevant).sum(axis=1)
#     coverage = coverage.filled(0)
#
#     return np.average(coverage, weights=sample_weight)

#------------------------------------------------------------------------------------------------#


def fbeta(y_true, y_pred, threshold_shift=0):
    """The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.
    """
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def precision(y_true, y_pred, threshold_shift=0):
    """ The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    """
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)

    return precision


def recall(y_true, y_pred, threshold_shift=0):
    """ The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    """
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    recall = tp / (tp + fn)

    return recall


def classification_report(y_true, y_pred, labels):
    import numpy
    from collections import Counter
    '''Similar to the one in sklearn.metrics, reports per classs recall, precision and F1 score'''
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('', 'recall', 'precision', 'f1-score', 'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report])
    N = len(y_true)
    print(formatter('avg / total', sum(report2[0]) / N, sum(report2[1]) / N, sum(report2[2]) / N, N) + '\n')


def coverage(test_dir,
             threshold= 0.75):
    import os
    import tqdm
    from predict import predict,image,preprocess_input

    img_path = os.listdir(test_dir)
    for image_path in tqdm(img_path):

        print image_path


