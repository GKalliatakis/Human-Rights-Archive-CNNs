from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

# from AUX_material.hra_baseline_class_weights import baseline_model
from AUX_material.hra_baseline_no_class_weights import baseline_model
from applications.hra_vgg16 import HRA_VGG16
from applications.hra_vgg19 import HRA_VGG19
from applications.hra_resnet50 import HRA_ResNet50
from applications.hra_vgg16_places365 import HRA_VGG16_Places365
from handcrafted_metrics import HRA_metrics, plot_confusion_matrix
from applications.compoundNet_vgg16 import CompoundNet_VGG16

# from applications.latest.hra_vgg16_checkpoint import HRA_VGG16
# from applications.latest.hra_vgg16_places365 import HRA_VGG16_Places365
# from applications.latest.compoundNet_vgg16_checkpoint import CompoundNet_VGG16

# ==== Baseline model ===========================================================================================================================
# model = baseline_model(classes=9, epochs=20, weights='HRA')
# model.summary()
# ===============================================================================================================================================


# ==== Feature extraction/Fine-tuing model ======================================================================================================
# pooling_mode = 'flatten'
# model = HRA_ResNet50(weights='HRA', mode='fine_tuning', pooling_mode=pooling_mode, include_top=True, data_augm_enabled=False)
# model.summary()
# ===============================================================================================================================================


# ==== CompoundNet model / Early-fusion==========================================================================================================
model= CompoundNet_VGG16(weights='HRA', mode= 'fine_tuning', fusion_strategy='average',  pooling_mode='avg', data_augm_enabled=False)
model.summary()
# ===============================================================================================================================================


# ==== Object-centric CompoundNet model =========================================================================================================
# model= CompoundNet_VGG16_VGG19(weights='HRA', mode= 'fine_tuning', fusion_strategy='maximum',  pooling_mode='max')
# model.summary()
# ===============================================================================================================================================


# ==== Late-fusion =========================================================================================================
# pooling_mode = 'max'

# model_a = HRA_VGG16(weights='HRA', mode='fine_tuning', pooling_mode='max', include_top=True, data_augm_enabled=False)
# model_a.summary()
#
# model_b = HRA_VGG16_Places365(weights='HRA', mode='fine_tuning', pooling_mode='flatten', include_top=True, data_augm_enabled=False)
# model_b.summary()

# ===============================================================================================================================================



model_name='CompoundNet_VGG16'



metrics = HRA_metrics(main_test_dir ='/home/sandbox/Desktop/Human_Rights_Archive_DB/test_uniform')

[y_true, y_pred, y_score] = metrics.predict_labels(model)

[y_true, y_pred] = metrics.duo_ensemble_predict_labels(model_a=model_a, model_b= model_b)


# print y_true
top1_acc = accuracy_score(y_true, y_pred)

# top5_acc = top_k_accuracy_score(y_true=y_true, y_pred=y_pred,k=3,normalize=True)
coverage = metrics.coverage(model,prob_threshold=0.85)
# coverage = metrics.coverage_duo_ensemble(model_a,model_b,prob_threshold=0.85)


# AP = average_precision_score (y_true = y_true, y_score=y_score)
#
# print AP



print ('\n')
print ('=======================================================================================================')
print (model_name+' Top-1 acc. =>  '+str(top1_acc))
print (model_name+' Coverage =>  '+str(coverage)+'%')



target_names = ['arms', 'child_labour', 'child_marriage', 'detention_centres', 'disability_rights', 'displaced_populations',
                        'environment', 'no_violation', 'out_of_school']

result= model_name+'  =>  '+ str(accuracy_score(y_true, y_pred))+ '\n'
result= model_name+'  =>  '+str(coverage)+'%'+ '\n'


f=open("results/coverage_late_fusion.txt", "a+")
f.write(result+'\n\n')
# f.write(str(y_pred)+'\n\n')
f.close()

print(classification_report(y_true, y_pred, target_names=target_names))

print (precision_score(y_true, y_pred, average=None))

cnf_matrix=confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


print (cnf_matrix.diagonal()/cnf_matrix.sum(axis=1))