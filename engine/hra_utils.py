"""Python utilities required by HRA."""

import json

import keras.backend as K
from keras.utils.data_utils import get_file


CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://github.com/GKalliatakis/expert-enigma/releases/download/0.6/HRA_class_index.json'


def hra_preprocess_input(x,
                         data_format=None,
                         scene_centric=False):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if scene_centric is True:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            x = x[:, ::-1, :, :]
            # Zero-center by mean pixel
            x[:, 0, :, :] -= 104.006
            x[:, 1, :, :] -= 116.669
            x[:, 2, :, :] -= 122.679
        else:
            # 'RGB'->'BGR'
            x = x[:, :, :, ::-1]
            # Zero-center by mean pixel
            x[:, :, :, 0] -= 104.006
            x[:, :, :, 1] -= 116.669
            x[:, :, :, 2] -= 122.679

    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            x = x[:, ::-1, :, :]
            # Zero-center by mean pixel
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
        else:
            # 'RGB'->'BGR'
            x = x[:, :, :, ::-1]
            # Zero-center by mean pixel
            x[:, :, :, 0] -= 103.939
            x[:, :, :, 1] -= 116.779
            x[:, :, :, 2] -= 123.68



    return x




def decode_predictions(preds, top=5):
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