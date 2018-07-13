"""Script for single image inference using the Human-Rights-Archive-CNNs.

    Example
    --------
    >>> python run_HRA_unified.py --img_path path/to/your/image/xxx.jpg --pre_trained_model VGG16 --pooling_mode avg --to_file output_filename.png

"""

from __future__ import print_function

import os



from utils.predict import *
from applications.hra_baseline import baseline_model
from applications.hra_resnet50 import HRA_ResNet50
from applications.hra_vgg16 import HRA_VGG16
from applications.hra_vgg19 import HRA_VGG19
from applications.hra_vgg16_places365 import HRA_VGG16_Places365

from visualisations.grad_cam import Grad_CAM


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type = str, help = 'path to image file')
    parser.add_argument("--pre_trained_model", type = str,help = 'One of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365` or `baseline`')
    parser.add_argument("--pooling_mode", type = str, help = 'One of `avg`, `max`, or `flatten`')
    parser.add_argument("--data_augm_enabled", type = bool, default = False, help = 'Whether to augment the samples during training or not')
    parser.add_argument("--nb_predictions", type=int, default=3,
                        help='Number of predictions returned by the model')
    parser.add_argument("--to_file", type=str, help='name to save the super imposed image to disk')

    args = parser.parse_args()
    return args

# model = baseline_model(classes=9, epochs=40, weights='HRA')

args = get_args()


if args.pre_trained_model == 'VGG16':
    model = HRA_VGG16(weights = 'HRA', mode = 'fine_tuning', pooling_mode = args.pooling_mode)
    layer2visualise = 'block5_conv3'

elif args.pre_trained_model == 'VGG19':
    model = HRA_VGG19(weights='HRA', mode='fine_tuning', pooling_mode=args.pooling_mode)
    layer2visualise = 'block5_conv3'

elif args.pre_trained_model == 'ResNet50':
    model = HRA_ResNet50(weights='HRA', mode='fine_tuning', pooling_mode=args.pooling_mode)

elif args.pre_trained_model == 'VGG16_Places365':
    model = HRA_VGG16_Places365(weights='HRA', mode='fine_tuning', pooling_mode=args.pooling_mode)
    layer2visualise = 'places_block5_conv3'

elif args.pre_trained_model == 'baseline':
    model = baseline_model(weights = 'HRA', epochs = 20)
    layer2visualise = 'block3_conv1'


img = image.load_img(args.img_path, target_size=(224, 224))
# preprocess image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)[0]
top_preds = np.argsort(preds)[::-1][0:args.nb_predictions]


superimposed_img = Grad_CAM(img_path=args.img_path,
                            model=model,
                            conv_layer_to_visualise=layer2visualise,
                            to_file=args.to_file)



# load the class label
file_name = 'categories_HRA.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://github.com/GKalliatakis/Human-Rights-Archive-CNNs/releases/download/v1.0/categories_HRA.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

print ('\n')
print('--PREDICTED HUMAN RIGHTS VIOLATIONS CATEGORIES:')
# output the prediction
for i in range(0, args.nb_predictions):
    print(classes[top_preds[i]], '->', preds[top_preds[i]])
