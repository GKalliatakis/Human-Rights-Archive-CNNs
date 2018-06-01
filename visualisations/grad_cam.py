# -*- coding: utf-8 -*-
"""Gradient-weighted Class Activation Mapping (Grad-CAM) for Keras.

# Reference:
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391v1.pdf)
"""

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def Grad_CAM(img_path,
             model,
             conv_layer_to_visualise= 'block5_conv3',
             to_file='test_Grad_CAM.jpg',
             ):
    """Produces heatmaps of "class activation" over input images.

        # Arguments
            img_path: the path to our target image.
            model: a trained Keras model instance
            layer2visualise: name of the convolutional layer from where the output feature map will be obtained
            correct_entry: the correct entry in the prediction vector of the model
            to_file: name to save the super imposed image to disk.

        # Returns
            The superimposed image.

        # Raises
            ValueError: in case of invalid argument for `weights`, or invalid input shape
    """

    img = image.load_img(img_path, target_size=(224, 224))
    # preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preprocessed_input = preprocess_input(x)

    preds = model.predict(x)
    inferred_entry=  (np.argmax(preds[0]))

    # This is the inferred entry in the prediction vector
    inferred_entry_output = model.output[:, inferred_entry]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer(conv_layer_to_visualise)

    # print last_conv_layer.get_config()

    # retrieve the filters shape for the selected layer
    filters_shape = last_conv_layer.filters


    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(inferred_entry_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([preprocessed_input])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(filters_shape):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()

    import cv2

    # We use cv2 to load the original image
    img = cv2.imread(img_path)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img

    # Save the image to disk
    cv2.imwrite(to_file, superimposed_img)

    return superimposed_img