import numpy as np
import matplotlib.pyplot as plt


def decode_predictions_basic(label,
        number_of_labels=5):
    """
    Returns a specified number of the best results in the label array

    # Arguments
            label: array, the labels
            number_of_labels: number of labels to be returned
    # Returns
            A list of the predicted labels
    """

    sorted_label = - np.sort(-label)
    sorted_label = np.ndarray.tolist(sorted_label)

    label = np.ndarray.tolist(label)

    labels = []

    for category in range(number_of_labels):
        probability = sorted_label[category]

        probability_index = label.index(probability)
        name_category = convert_index2name(probability_index)
        labels.append(name_category + " : " + str(probability))
    return labels


def convert_index2name(index):

    """
    Converts an integer into the corresponding category

    # Arguments
            index: integer between 0 and number_of_categories - 1

    # Returns
            Name of the category
    """

    if index == 0:
        return "Arms"
    elif index ==1:
        return "ChildLabour"
    elif index == 2:
        return "ChildMarriage"
    elif index == 3:
        return "DetentionCentres"
    elif index == 4:
        return "Disability"
    elif index == 5:
        return "Environment"
    elif index == 6:
        return "NoViolation"
    elif index == 7:
        return "OutOfSchool"
    elif index == 8:
        return "Refugees"
    else:
        raise Exception("Wrong length :", index)



def top3(labels):
    first_prediction = labels[0:1]
    second_prediction = labels[1:2]
    third_prediction = labels[2:3]

    first_prediction_list = ', '.join(first_prediction)
    second_prediction_list = ', '.join(second_prediction)
    third_prediction_list = ', '.join(third_prediction)

    first_prediction_tmp = first_prediction_list.split(' ')
    second_prediction_tmp = second_prediction_list.split(' ')
    third_prediction_tmp = third_prediction_list.split(' ')

    first_prediction_label = first_prediction_tmp[0]
    second_prediction_label = second_prediction_tmp[0]
    third_prediction_label = third_prediction_tmp[0]

    first_prediction_rounded_accuracy = round(float(first_prediction_tmp[2]), 3)
    second_prediction_rounded_accuracy = round(float(second_prediction_tmp[2]), 3)
    third_prediction_rounded_accuracy = round(float(third_prediction_tmp[2]), 3)

    first_prediction__rounded_accuracy = str(first_prediction_rounded_accuracy)
    second_prediction__rounded_accuracy = str(second_prediction_rounded_accuracy)
    third_prediction__rounded_accuracy = str(third_prediction_rounded_accuracy)

    final_predictions_pt1 = first_prediction_label + ' ' + first_prediction__rounded_accuracy + ', ' + second_prediction_label + ' ' + second_prediction__rounded_accuracy + ', ' + third_prediction_label + ' ' + third_prediction__rounded_accuracy
    #final_predictions_pt2 = 'Predicted human rights violations categories: '
    #print 'Predicted human rights violations categories: ' + first_prediction_label + ' ' + first_prediction__rounded_accuracy + ', ' + second_prediction_label + ' ' + second_prediction__rounded_accuracy + ', ' + third_prediction_label + ' ' + third_prediction__rounded_accuracy

    return final_predictions_pt1


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """

  order = list(reversed(range(len(preds))))
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("Arms", "ChildLabour", "ChildMarriage", "DetentionCentres", "Disability", "Environment", "NoViolation", "OutOfSchool", "Refugees")
  plt.barh(order, preds, alpha=0.65)
  plt.yticks(order, labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()









