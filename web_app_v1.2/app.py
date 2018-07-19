import os
# scientific computing library for saving, reading, and resizing images
# for matrix math
# for importing our keras model
# for regular expressions, saves time dealing with string data
import re

# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request

import tensorflow as tf
from flask_uploads import UploadSet, configure_uploads, IMAGES

# from utils.predict import *
from hra_vgg19 import HRA_VGG19
from hra_utils import *

from PIL import Image
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

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

        #----------- Code to predict HRA category -----------
        # Code to predict HRA category

        with graph.as_default():

            model = HRA_VGG19(weights='HRA', mode='fine_tuning', pooling_mode='avg')

            img = image.load_img(image_path, target_size=(224, 224))
            # preprocess image
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)[0]

            list_label = decode_predictions_basic(preds, number_of_labels=5)
            top3_results = top3(labels=list_label)


            im1 = Image.open(image_path)
            # adjust width and height to your needs
            width = 350
            height = 350
            im1 = im1.resize((width, height), Image.ANTIALIAS)  # best down-sizing filter

            im1.save(image_path)

            return render_template('index.html', final_predictions_pt1=top3_results, image=image_path)


if __name__ == "__main__":
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    app.run(host='0.0.0.0', port=port,debug=True)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
