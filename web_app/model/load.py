import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

from keras.applications.resnet50 import ResNet50



def init():



    # load json and create model
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load weights into new model
    loaded_model.load_weights("model.h5")

    # print("Loaded Model from disk")

    #compile and evaluate loaded model
    # loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    loaded_model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])



    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return loaded_model,graph