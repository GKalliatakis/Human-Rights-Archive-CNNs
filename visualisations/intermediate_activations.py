from python.Learning_Image_Representations_for_Recognising_HRV.examples.master.predict import *
from python.Learning_Image_Representations_for_Recognising_HRV.examples.master.hra_feature_transfer import transfer_learning,fine_tune
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models



# # for server
# pre_trained_model = sys.argv[1]


# for local pc
pre_trained_model = 'VGG16'

transfer_learning_model = transfer_learning(pre_trained_model=pre_trained_model,
                                            weights='HRA',
                                            classes=9,
                                            augmented_samples=False)

# transfer_learning_model.summary()
#
# fine_tuned_model = fine_tune(transfer_learning_model,
#                              pre_trained_model=pre_trained_model,
#                              weights='HRA',
#                              classes=9,
#                              augmented_samples=False)
#
# fine_tuned_model.summary()


img_path = '/home/sandbox/Desktop/Testing Images/human_right_viol_2.jpg'


img = image.load_img('/home/sandbox/Desktop/Testing Images/human_right_viol_4.jpg', target_size=(224, 224))

img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in transfer_learning_model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=transfer_learning_model.input, outputs=layer_outputs)

# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[1]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()





# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in transfer_learning_model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
