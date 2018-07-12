#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
#for matrix math
#for importing our keras model
#for regular expressions, saves time dealing with string data
import re

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
# for reading operating system data
import os
# scientific computing library for saving, reading, and resizing images
# for matrix math
# for importing our keras model
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys

# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request

#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
#from load import *

from model.load import *


from flask_uploads import UploadSet, configure_uploads, IMAGES

# from python.Learning_Image_Representations_for_Recognising_HRV.examples.master.hra_transfer_cnn import fine_tune
from utils.predict import *
from visualisations.grad_cam import Grad_CAM
from applications.hra_vgg16 import HRA_VGG16
from applications.hra_vgg16_places365 import HRA_VGG16_Places365
from hra_utils import *


from PIL import Image

#initalize the flask app
app = Flask(__name__)


photos = UploadSet('photos', IMAGES)


app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


#global vars for easy reusability
global model, graph

#initialize these variables
# model, graph = init()

graph = tf.get_default_graph()

#decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)',imgData1).group(1)
    #print(imgstr)
    with open('output.png','wb') as output:
        output.write(imgstr.decode('base64'))


@app.route('/')
def index():
    #initModel()
    #render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])

        image_path = './static/img/' + filename
        # image_path = '/home/user/Human-Rights-Archive-CNNs/web_app/static/img/' + filename

        #----------- Code to predict HRA category -----------
        # Code to predict HRA category

        with graph.as_default():

            model = HRA_VGG16(weights='HRA', mode='fine_tuning', pooling_mode='avg')

            img = image.load_img(image_path, target_size=(224, 224))

            # preprocess image
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)[0]

            list_label = decode_predictions_basic(preds, number_of_labels=5)
            top3_results = top3(labels=list_label)

            print top3_results

            top_preds = np.argsort(preds)[::-1][0:10]

            # load the class label
            # file_name = '/home/user/Human-Rights-Archive-CNNs/web_app/categories_HRA.txt'
            file_name = '/home/sandbox/Desktop/categories_HRA.txt'
            classes = list()
            with open(file_name) as class_file:
                for line in class_file:
                    classes.append(line.strip().split(' ')[0][3:])
            classes = tuple(classes)

            print('--HUMAN RIGHTS VIOLATIONS CATEGORIES:')
            # output the prediction
            for i in range(0, 5):
                print(classes[top_preds[i]])




            return render_template('index.html', final_predictions_pt1=top3_results, image=image_path)


if __name__ == "__main__":
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    app.run(host='0.0.0.0', port=port,debug=True)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
