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
from hra_utils import *


import PIL
from PIL import Image

#initalize the flask app
app = Flask(__name__)


photos = UploadSet('photos', IMAGES)


app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)




#global vars for easy reusability
global model, graph

#initialize these variables
model, graph = init()

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
        Grad_CAM_image_path = './static/img/Grad_cam' + filename


        #----------- Code to predict HRA category -----------
        # Code to predict HRA category

        with graph.as_default():

            # model = fine_tune(pre_trained_model='VGG16',
            #                   classes=9,
            #                   augmented_samples=False,
            #                   weights='HRA', weights_top_layers='HRA')


            # model = HRA_VGG16(weights='HRA', mode='FT', pooling='avg' ,include_top=True)
            model = HRA_VGG16(weights='HRA', mode='fine_tuning', pooling_mode='avg')

            # model = transfer_learning(pre_trained_model='VGG16',
            #                                    weights='HRA',
            #                                    classes=9,
            #                                    augmented_samples=False)

            img = image.load_img(image_path, target_size=(224, 224))
            # preprocess image
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)


            Grad_CAM(image_path,
                     model=model,
                     conv_layer_to_visualise='block5_conv3',
                     to_file=Grad_CAM_image_path)


            # preds = predict(model, img, target_size)
            # preds= HRA_predict(model, img, target_size, top_n=3)
            # predicted = np.argmax(preds)
            # print preds

            preds = model.predict(x)[0]

            list_label = decode_predictions_basic(preds, number_of_labels=5)
            top3_results = top3(labels=list_label)

            print top3_results

            top_preds = np.argsort(preds)[::-1][0:10]

            # load the class label
            file_name = 'categories_HRA.txt'
            classes = list()
            with open(file_name) as class_file:
                for line in class_file:
                    classes.append(line.strip().split(' ')[0][3:])
            classes = tuple(classes)

            print('--HUMAN RIGHTS VIOLATIONS CATEGORIES:')
            # output the prediction
            for i in range(0, 5):
                print(classes[top_preds[i]])

            basewidth = 550
            img = Image.open(Grad_CAM_image_path)
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
            # tmp = Grad_CAM_image_path
            img.save(Grad_CAM_image_path)


            return render_template('index.html', final_predictions_pt1=top3_results, image=Grad_CAM_image_path)

        #return redirect(url_for('index'))



# @app.route('/predict/',methods=['GET','POST'])
# def predict():
#     #whenever the predict method is called, we're going
#     #to input the user drawn character as an image into the model
#     #perform inference, and return the classification
#     #get the raw data format of the image
#     imgData = request.get_data()
#     #encode it into a suitable format
#     #convertImage(imgData)
#     print "debug"
#     #read the image into memory
#     #x = imread('output.png',mode='L')
#
#     x = imread('output.png')
#
#     # x = image.load_img(PATH_IMAGE, target_size=(224, 224))
#
#
#     #compute a bit-wise inversion so black becomes white and vice versa
#     #x = np.invert(x)
#
#     #make it the right size
#     x = imresize(x,(224,224))
#     imshow(x)
#
#     #convert to a 4D tensor to feed into our model
#     x = x.reshape(1,224,224,1)
#     print "debug2"
#     #in our computation graph
#
#     with graph.as_default():
#         #perform the prediction
#         out = model.predict(x)
#         print(out)
#         print(np.argmax(out,axis=1))
#         print "debug3"
#         #convert the response to a string
#         response = np.array_str(np.argmax(out,axis=1))
#         return response



#
# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST' and 'photo' in request.files:
#         filename = photos.save(request.files['photo'])
#         image_path = './static/img/' + filename
#
#
#         #----------- Code to predict HRA category -----------
#         # Code to predict HRA category
#
#         with graph.as_default():
#
#             Y = HumanRightsViolations_classifier(type_model='RN2L', architecture_model=[9])
#             #Y = HumanRightsViolations_classifier(type_model='VGG19_1L', architecture_model=[9])
#
#
#
#             result = Y.label(image_path)
#
#             print result
#             # print '*********************************'
#
#             first_prediction = result[0:1]
#             second_prediction= result [1:2]
#             third_prediction=result [2:3]
#
#
#
#             first_prediction_list= ', '.join(first_prediction)
#             second_prediction_list = ', '.join(second_prediction)
#             third_prediction_list = ', '.join(third_prediction)
#
#             first_prediction_tmp = first_prediction_list.split(' ')
#             second_prediction_tmp = second_prediction_list.split(' ')
#             third_prediction_tmp = third_prediction_list.split(' ')
#
#             first_prediction_label = first_prediction_tmp[0]
#             second_prediction_label = second_prediction_tmp[0]
#             third_prediction_label = third_prediction_tmp[0]
#
#             first_prediction_rounded_accuracy = round(float(first_prediction_tmp[2]), 3)
#             second_prediction_rounded_accuracy = round(float(second_prediction_tmp[2]), 3)
#             third_prediction_rounded_accuracy = round(float(third_prediction_tmp[2]), 3)
#
#             first_prediction__rounded_accuracy = str(first_prediction_rounded_accuracy)
#             second_prediction__rounded_accuracy = str(second_prediction_rounded_accuracy)
#             third_prediction__rounded_accuracy = str(third_prediction_rounded_accuracy)
#
#             final_predictions_pt1=first_prediction_label+' '+first_prediction__rounded_accuracy +', '+second_prediction_label+' '+second_prediction__rounded_accuracy+', '+third_prediction_label+' '+third_prediction__rounded_accuracy
#             final_predictions_pt2='Predicted human rights violations categories: '
#             print 'Predicted human rights violations categories: ' +first_prediction_label+' '+first_prediction__rounded_accuracy +', '+second_prediction_label+' '+second_prediction__rounded_accuracy+', '+third_prediction_label+' '+third_prediction__rounded_accuracy
#             # print '1st: ' +first_prediction_label+' '+first_prediction__rounded_accuracy
#             # print '2nd: ' + second_prediction_label + ' ' + second_prediction__rounded_accuracy
#             # print '3rd: ' + third_prediction_label + ' ' + third_prediction__rounded_accuracy
#
#
#
#             # print result
#
#     # return first_prediction_label
#
#
#     basewidth = 400
#     img= Image.open(image_path)
#     wpercent = (basewidth / float(img.size[0]))
#     hsize = int((float(img.size[1]) * float(wpercent)))
#     img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
#     tmp='./static/img/' + filename
#     img.save(tmp)
#
#     return render_template('index.html', final_predictions_pt1=final_predictions_pt1, image=tmp, final_predictions_pt2=final_predictions_pt2)




if __name__ == "__main__":
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    app.run(host='0.0.0.0', port=port,debug=True)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
