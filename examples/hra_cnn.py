# -*- coding: utf-8 -*-
"""Trains a simple convnet on the Human Rights Archive (HRA) dataset.
"""

from applications.latest.hra_vgg19 import HRA_VGG19

from utils.predict import *

# model = baseline_model(classes=9, epochs=40, weights='HRA')


pooling_mode = 'avg'
model = HRA_VGG19(weights='HRA', mode='fine_tuning', pooling_mode=pooling_mode, include_top=True, data_augm_enabled=False)
model.summary()
model.summary()

### Simple prediction example ###
img = image.load_img('/home/sandbox/Desktop/Human_Rights_Archive_DB/test_uniform/out_of_school/out_of_school_0017.jpg', target_size=(224, 224))
# preprocess image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = predict(model, img, target_size)
predicted = np.argmax(preds)

print preds
# plot_preds(img, preds)

### Integrated Gradients ###
# ig = integrated_gradients(model)
#
#
# # preprocess reference as well
# ref = np.zeros((224, 224, 3))
# ref = np.expand_dims(ref, axis=0)
# ref = preprocess_input(ref)
#
# exp = ig.explain(x[0], reference=ref[0], outc=predicted)
#
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
