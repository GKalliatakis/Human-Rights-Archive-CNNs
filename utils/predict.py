import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

from keras.utils import get_file
from keras import backend as K
import json


target_size = (224, 224)

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://github.com/GKalliatakis/expert-enigma/releases/download/0.7/HRA_class_index.json'


def decode_predictions(preds, top=3):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: integer, how many top-guesses to return.

    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: in case of invalid shape of the `pred` array
            (must be 2D).
    """
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 9:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 9)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('HRA_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='hra_models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def predict(model, img, target_size, top_n=9):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds, top=top_n)[0]



def duo_ensemble_predict(model_a, model_b,
                         img,
                         target_size,
                         top_n=9):
  """Pool the predictions of a set of classifiers (to ensemble the classifiers) is to average their predictions at inference time
  Args:
    model_a: 1st model
    model_b: 2nd model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels (which have been pooled accordingly) and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds_a = model_a.predict(x)
  preds_b = model_b.predict(x)
  final_preds = 0.50 * (preds_a + preds_b)

  return decode_predictions(final_preds,top=top_n)[0]



def trio_ensemble_predict(model_a, model_b, model_c,
                          img,
                          target_size,
                          top_n=9):
  """Pool the predictions of a set of classifiers (to ensemble the classifiers) is to average their predictions at inference time
  Args:
    model_a: 1st model
    model_b: 2nd model
    model_c: 3rd model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels (which have been pooled accordingly) and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds_a = model_a.predict(x)
  preds_b = model_b.predict(x)
  preds_c = model_c.predict(x)
  final_preds = 0.333 * (preds_a + preds_b + preds_c)

  return decode_predictions(final_preds,top=top_n)[0]


def quadruple_ensemble_predict(model_a, model_b, model_c, model_d,
                               img,
                               target_size,
                               top_n=9):
    """Pool the predictions of a set of classifiers (to ensemble the classifiers) is to average their predictions at inference time
    Args:
      model_a: 1st model
      model_b: 2nd model
      model_c: 3rd model
      img: PIL format image
      target_size: (w,h) tuple
      top_n: # of top predictions to return
    Returns:
      list of predicted labels (which have been pooled accordingly) and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds_a = model_a.predict(x)
    preds_b = model_b.predict(x)
    preds_c = model_c.predict(x)
    preds_d = model_d.predict(x)
    final_preds = 0.25 * (preds_a + preds_b + preds_c + preds_d)

    return decode_predictions(final_preds, top=top_n)[0]


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """

  plt.figure()
  plt.subplot(1, 2, 1)
  plt.imshow(image)
  plt.title("Original image")




  # plt.axis('off')

  # plt.figure()



  # labels = ('Arms', 'ChildLabour', 'ChildMarriage', 'DetentionCentres', 'Disability', 'Environment',
  #                         'NoViolation', 'OutOfSchool', 'Refugees')
  # plt.barh([0, 1, 2], preds, alpha=0.5)
  # plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], labels)
  # plt.xlabel('Probability')
  # plt.xlim(0,1.01)
  # plt.tight_layout()
  # plt.show()


  plt.subplot(1, 2, 2)
  order = list(reversed(range(len(preds))))
  bar_preds = [pr[2] for pr in preds]
  labels = (pr[1] for pr in preds)
  plt.barh(order, bar_preds, alpha=0.5)
  plt.yticks(order, labels)
  plt.title("Predicted Probabilities")
  plt.xlabel('Probability')
  plt.ylabel('Classes')
  plt.xlim(0, 1.01)
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.2)
  plt.show()


# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.title("Original image")
#
# th = max(np.abs(np.min(exp)), np.abs(np.max(exp)))
# plt.subplot(1, 2, 2)
# plt.imshow(np.sum(exp, axis=2), cmap="seismic", vmin=-1 * th, vmax=th)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.title("Explanation")
# plt.show()

