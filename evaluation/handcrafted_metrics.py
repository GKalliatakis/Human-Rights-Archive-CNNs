import os
from utils.predict import *
import itertools



class HRA_metrics():
    """Perfofmance metrics base class.
    """
    def __init__(self,
                 main_test_dir ='/home/sandbox/Desktop/Human_Rights_Archive_DB/test'
                 ):


        self.main_test_dir = main_test_dir
        self.total_nb_of_test_images = sum([len(files) for r, d, files in os.walk(main_test_dir)])
        self.sorted_categories_names = sorted(os.listdir(main_test_dir))
        self.support = [ 186, 945, 87, 186, 273, 408, 201, 153, 609]



    def predict_labels(self,
                       model):
        """Computes the predicted and ground truth labels, as returned by a single classifier.

            # Arguments
                model = keras model for which we want to predict the labels.

            # Returns
                y_true : 1d array-like containing the ground truth (correct) labels.
                y_pred : 1d array-like containing the predicted labels, as returned by a classifier.
        """
        y_pred = []
        y_true= []
        y_score = []

        for hra_class in self.sorted_categories_names:

            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, hra_class)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:

                if hra_class == 'arms':
                    true_label = 0
                elif hra_class == 'child_labour':
                    true_label = 1
                elif hra_class == 'child_marriage':
                    true_label = 2
                elif hra_class == 'detention_centres':
                    true_label = 3
                elif hra_class == 'disability_rights':
                    true_label = 4
                elif hra_class == 'displaced_populations':
                    true_label = 5
                elif hra_class == 'environment':
                    true_label = 6
                elif hra_class == 'no_violation':
                    true_label = 7
                elif hra_class == 'out_of_school':
                    true_label = 8

                y_true.append(true_label)


                # variable that contains the final image to be loaded
                print ('Processing  [' + raw_img + ']')
                final_img = os.path.join(tmp, raw_img)
                img = image.load_img(final_img, target_size=(224, 224))

                preds = predict(model, img, target_size)

                y_pred.append(int(preds[0][0]))
                y_score.append(int(preds[0][2]))


        print y_pred

        return y_true, y_pred, y_score



    def predict_labels_KNeighborsClassifier(self,
                                            classifier):
        """Computes the predicted and ground truth labels, as returned by a single classifier.

            # Arguments
                model = keras model for which we want to predict the labels.

            # Returns
                y_true : 1d array-like containing the ground truth (correct) labels.
                y_pred : 1d array-like containing the predicted labels, as returned by a classifier.
        """
        y_pred = []
        y_true= []
        y_score = []

        for hra_class in self.sorted_categories_names:

            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, hra_class)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:

                if hra_class == 'arms':
                    true_label = 0
                elif hra_class == 'child_labour':
                    true_label = 1
                elif hra_class == 'child_marriage':
                    true_label = 2
                elif hra_class == 'detention_centres':
                    true_label = 3
                elif hra_class == 'disability_rights':
                    true_label = 4
                elif hra_class == 'displaced_populations':
                    true_label = 5
                elif hra_class == 'environment':
                    true_label = 6
                elif hra_class == 'no_violation':
                    true_label = 7
                elif hra_class == 'out_of_school':
                    true_label = 8

                y_true.append(true_label)


                # variable that contains the final image to be loaded
                print ('Processing  [' + raw_img + ']')
                final_img = os.path.join(tmp, raw_img)
                img = image.load_img(final_img, target_size=(224, 224))

                preds = predict(model, img, target_size)

                y_pred.append(int(preds[0][0]))
                y_score.append(int(preds[0][2]))


        print y_pred

        return y_true, y_pred, y_score



    def duo_ensemble_predict_labels(self,
                                    model_a,
                                    model_b):
        """Computes the predicted and ground truth labels, as returned by an ansemble of 2 classifiers.

        # Arguments
            model_a: 1st model
            model_b: 2nd model

        # Returns
            y_true : 1d array-like containing the ground truth (correct) labels.
            y_pred : 1d array-like containing the predicted labels, as returned by a classifier.
        """

        y_pred = []
        y_true = []

        for hra_class in self.sorted_categories_names:
            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, hra_class)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:

                if hra_class == 'arms':
                    true_label = 0
                elif hra_class == 'child_labour':
                    true_label = 1
                elif hra_class == 'child_marriage':
                    true_label = 2
                elif hra_class == 'detention_centres':
                    true_label = 3
                elif hra_class == 'disability_rights':
                    true_label = 4
                elif hra_class == 'displaced_populations':
                    true_label = 5
                elif hra_class == 'environment':
                    true_label = 6
                elif hra_class == 'no_violation':
                    true_label = 7
                elif hra_class == 'out_of_school':
                    true_label = 8

                y_true.append(true_label)

                # variable that contains the final image to be loaded
                print ('Processing  [' + raw_img + ']')
                final_img = os.path.join(tmp, raw_img)
                img = image.load_img(final_img, target_size=(224, 224))

                preds = duo_ensemble_predict(model_a, model_b, img, target_size)

                y_pred.append(int(preds[0][0]))

        print y_pred

        return y_true, y_pred



    def trio_ensemble_predict_labels(self,
                                     model_a,
                                     model_b,
                                     model_c):
        """Computes the predicted and ground truth labels, as returned by an ansemble of 3 classifiers.

        # Arguments
            model_a: 1st model
            model_b: 2nd model
            model_c: 3rd model

        # Returns
            y_true : 1d array-like containing the ground truth (correct) labels.
            y_pred : 1d array-like containing the predicted labels, as returned by a classifier.
        """

        y_pred = []
        y_true = []

        for hra_class in self.sorted_categories_names:
            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, hra_class)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:

                if hra_class == 'arms':
                    true_label = 0
                elif hra_class == 'child_labour':
                    true_label = 1
                elif hra_class == 'child_marriage':
                    true_label = 2
                elif hra_class == 'detention_centres':
                    true_label = 3
                elif hra_class == 'disability_rights':
                    true_label = 4
                elif hra_class == 'displaced_populations':
                    true_label = 5
                elif hra_class == 'environment':
                    true_label = 6
                elif hra_class == 'no_violation':
                    true_label = 7
                elif hra_class == 'out_of_school':
                    true_label = 8

                y_true.append(true_label)

                # variable that contains the final image to be loaded
                print ('Processing  [' + raw_img + ']')
                final_img = os.path.join(tmp, raw_img)
                img = image.load_img(final_img, target_size=(224, 224))

                preds = trio_ensemble_predict(model_a, model_b,model_c, img, target_size)

                y_pred.append(int(preds[0][0]))

        print y_pred

        return y_true, y_pred



    def quadruple_ensemble_predict_labels(self,
                                          model_a,
                                          model_b,
                                          model_c,
                                          model_d):
        """Computes the predicted and ground truth labels, as returned by an ansemble of 4 classifiers.

        # Arguments
            model_a: 1st model
            model_b: 2nd model
            model_c: 3rd model
            model_d: 4th model

        # Returns
            y_true : 1d array-like containing the ground truth (correct) labels.
            y_pred : 1d array-like containing the predicted labels, as returned by a classifier.
        """

        y_pred = []
        y_true = []

        for hra_class in self.sorted_categories_names:
            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, hra_class)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:

                if hra_class == 'arms':
                    true_label = 0
                elif hra_class == 'child_labour':
                    true_label = 1
                elif hra_class == 'child_marriage':
                    true_label = 2
                elif hra_class == 'detention_centres':
                    true_label = 3
                elif hra_class == 'disability_rights':
                    true_label = 4
                elif hra_class == 'displaced_populations':
                    true_label = 5
                elif hra_class == 'environment':
                    true_label = 6
                elif hra_class == 'no_violation':
                    true_label = 7
                elif hra_class == 'out_of_school':
                    true_label = 8

                y_true.append(true_label)

                # variable that contains the final image to be loaded
                print ('Processing  [' + raw_img + ']')
                final_img = os.path.join(tmp, raw_img)
                img = image.load_img(final_img, target_size=(224, 224))

                preds = quadruple_ensemble_predict(model_a, model_b,model_c,model_d, img, target_size)

                y_pred.append(int(preds[0][0]))

        print y_pred

        return y_true, y_pred



    def coverage(self,
                 model,
                 prob_threshold = 0.75):
        """Coverage is the fraction of examples for which the ML system is able to produce a response.
        """


        predicted_class_list = []
        actual_class_list = []
        coverage_count = 0

        for category in self.sorted_categories_names:
            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, category)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:
                # variable that contains the final image to be loaded
                final_img = os.path.join(tmp, raw_img)
                img = image.load_img(final_img, target_size=(224, 224))
                # preprocess image
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = predict(model, img, target_size)

                top_1_predicted_probability = preds[0][2]

                # top_1_predicted = np.argmax(preds)
                top_1_predicted_label = preds[0][1]

                if top_1_predicted_probability >= prob_threshold:
                    coverage_count += 1

                print ('`' + category + '/' + raw_img + '`  ===>  `' +
                       top_1_predicted_label + '`' + ' with ' + str(top_1_predicted_probability) + ' P')

                predicted_class_list.append(top_1_predicted_label)
                actual_class_list.append(category)

        total_coverage_per = (coverage_count * 100) / self.total_nb_of_test_images

        return total_coverage_per


    def coverage_duo_ensemble(self,
                              model_a,
                              model_b,
                              prob_threshold = 0.75):
        """Coverage is the fraction of examples for which the ML system is able to produce a response.
        """


        predicted_class_list = []
        actual_class_list = []
        coverage_count = 0

        for category in self.sorted_categories_names:
            # variable that contains the main dir alongside the selected category
            tmp = os.path.join(self.main_test_dir, category)
            img_names = sorted(os.listdir(tmp))

            for raw_img in img_names:
                # variable that contains the final image to be loaded
                final_img = os.path.join(tmp, raw_img)
                img = image.load_img(final_img, target_size=(224, 224))
                # preprocess image
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = duo_ensemble_predict(model_a, model_b, img, target_size)
                # preds = predict(model, img, target_size)

                top_1_predicted_probability = preds[0][2]

                # top_1_predicted = np.argmax(preds)
                top_1_predicted_label = preds[0][1]

                if top_1_predicted_probability >= prob_threshold:
                    coverage_count += 1

                print ('`' + category + '/' + raw_img + '`  ===>  `' +
                       top_1_predicted_label + '`' + ' with ' + str(top_1_predicted_probability) + ' P')

                predicted_class_list.append(top_1_predicted_label)
                actual_class_list.append(category)

        total_coverage_per = (coverage_count * 100) / self.total_nb_of_test_images

        return total_coverage_per

def top_k_accuracy_score(y_true, y_pred, k=5, normalize=True):
    """Top k Accuracy classification score.
    For multiclass classification tasks, this metric returns the
    number of times that the correct class was among the top k classes
    predicted.
    Parameters
    ----------
    y_true : 1d array-like, or class indicator array / sparse matrix
        shape num_samples or [num_samples, num_classes]
        Ground truth (correct) classes.
    y_pred : array-like, shape [num_samples, num_classes]
        For each sample, each row represents the
        likelihood of each possible class.
        The number of columns must be at least as large as the set of possible
        classes.
    k : int, optional (default=5) predictions are counted as correct if
        probability of correct class is in the top k classes.
    normalize : bool, optional (default=True)
        If ``False``, return the number of top k correctly classified samples.
        Otherwise, return the fraction of top k correctly classified samples.
    Returns
    -------
    score : float
        If ``normalize == True``, return the proportion of top k correctly
        classified samples, (float), else it returns the number of top k
        correctly classified samples (int.)
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    See also
    --------
    accuracy_score
    Notes
    -----
    If k = 1, the result will be the same as the accuracy_score (though see
    note below). If k is the same as the number of classes, this score will be
    perfect and meaningless.
    In cases where two or more classes are assigned equal likelihood, the
    result may be incorrect if one of those classes falls at the threshold, as
    one class must be chosen to be the nth class and the class chosen may not
    be the correct one.
    """
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx + 1:]:
            counter += 1
    if normalize:
        return counter / num_obs
    else:
        return counter



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
