#!/usr/bin/env python
# coding: utf-8

# Multiclass, multipatient segmentation method for DICOM format input CT data and multiclass mask files
# Structured U-net for Sensitivity Analysis (Last modified: 1/7/2022) Author: Chris Boyd
"""TO DO:
    Add Sphinx for documentation
    Check var_limit on A.GaussNoise
"""
#
# Significant recent changes:
#         Remove sklearn metrics and add overall evaluation via Dice/Jaccard
#         Improve iterative recording and display of split and class metrics
#         Clear cache during training
#         Change zoom to crop
#         Drop slices where sum of image < minimum
#         Added learning rate scheduler and early_stop callbacks
#         Leaky GPU Ram issue fixed as optimisers were retained in memory with each split
#
# Continue:
#     Repair of reference image/mask/prediction at end of code
#
#
#  Adapted from:https://github.com/fsan/ptc5750_ctscan_segmentation_with_cnn
#            with assistance from Dr. Wolfgang Mayer, A/Prof. Mark McDonnell & Prof Mark Jenkinson
#%%
import os, sys
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
checkpoint_dir = os.path.join(base_dir, 'checkpoints')
print(checkpoint_dir)

from IRCAD.utils import (gpu_check, pixel_count, predict_masks, plot_image_and_masks)
from IRCAD.DataLoader import PatientData
from IRCAD.training import lpo_train, sss_train, kfold_train
from itertools import product
from IRCAD.metrics_tf import metric_tf_initialise
from models.custom_unet import custom_unet
import numpy as np
import pandas as pd

#%%
#Define model parameters, including list of parameters for iteration - Based on: https://github.com/autonomio/talos & https://autonomio.github.io/talos/#/

params = {'root_dir':("D:\\PhD\\Programming Practice\\Keras\\Sensitivity\\PatientData"),
          'patient_folder_string': ("3Dircadb"),
          'organs': ['bone', 'liver'],
          'init_sample': (15),
          'training_sample': (10), #5|10|15 - Look at data ?10|12|15
          'patch_size':(128), #32|64|128
          'filters':(32), #[16,32,64], #Run 2pt trial run - [OOM: 128]
          'dropout':[0.5, 0.3, 0.1], #Trial run
          'batches': (16),#(32),
          'opt_name': ["Adam", "RMSprop"], #"Adagrad", "SGD",
          'loss_fn_name': ("Recall Favoured"),#"Combo Loss"],# "Focal Tversky Loss"], # "Dice Loss","Jaccard Distance", 
          'epochs': (10), #check epoch of best model in current data
          'crossval_method': ("lpo"),#["kfold", "lpo", "sss"],
          'p-out':(1),
          'test_size':(3/10),
          'number_splits':(5),
          'num layers':(4),
          'import_source': ('dicom'),#'dicom' or 'npz'
          }#[3,4,5]}

#Confirm tensorflow will be used
gpu_check()

#Count relative occurance of classes in image files

#pixel_count(params['root_dir'], filetype='', organ_string=params['organs'])

#%% Import patient data and binary mask files
if params['import_source'] == 'dicom':
    patient_paths = PatientData.patient_dirs(params['root_dir'], params['patient_folder_string'])
    input_allPt, target_allPt = PatientData.pt_mask_import(params['init_sample'], params['organs'],
                                                           patient_paths, figures=False, augment=False, patchsize=params['patch_size'])
elif params['import_source'] == 'npz':
    archive_path = r"D:\PhD\Programming Practice\Keras\Sensitivity\Sensitivity\IRCAD\checkpoints\2organ-2copies-10patients-aug_data.npz"
    input_allPt = np.load(archive_path, allow_pickle = True)['arr_0']
    target_allPt = np.load(archive_path, allow_pickle = True)['arr_1']

#%% Iteratively and combinatorially training k-fold validated models

input_trainingPt, target_trainingPt = input_allPt[:params['training_sample']], target_allPt[:params['training_sample']]

validation_length = params['training_sample']-params['init_sample']
input_validationPt, target_validationPt = (input_allPt[validation_length:], target_allPt[validation_length:])

# Define model architecture parameters
num_classes = target_allPt[0].shape[-1]

# Create model
input_shape = (params['patch_size'], params['patch_size'], 1)


crossval_method = params['crossval_method']

#%% Iteratively and combinatorially training models with validation specified in params dictionary
for optim, dropout  in product(params['opt_name'], params['dropout']):

    run_string = (str(params['training_sample'])+"_"+ str(params['patch_size'])+"_"+ str(params['filters'])+"_"+str(dropout)+
                  "_"+str(params['batches'])+"_"+str(params['num layers'])+"_"+str(optim)+"_"+str(params['loss_fn_name']))
    run_checkpoint_dir = os.path.join(checkpoint_dir, run_string)

    # Train model
    if crossval_method == "lpo":

        lpo_train(input_trainingPt, target_trainingPt, input_shape, num_classes, params['filters'], dropout,
                  params['num layers'], run_checkpoint_dir, optim, params['loss_fn_name'], batches=params['batches'],
                  epochs = params['epochs'], print_summary=False, p_out=params['p-out'])

    elif crossval_method == "sss":

        sss_train(input_trainingPt, target_trainingPt, input_shape, num_classes, params['filters'], params['dropout'],
                  params['num layers'], run_checkpoint_dir, optim, params['loss_fn_name'], path_string_list = patient_paths,
                  training_length=params['training_sample'],  batches=params['batches'], epochs = params['epochs'], print_summary=False,
                  test_size=params['test_size'], n_splits=params['number_splits'], random_state=42)

    elif crossval_method == "kfold":

        kfold_train(input_trainingPt, target_trainingPt, input_shape, num_classes, params['filters'], params['dropout'],
                  params['num layers'], run_checkpoint_dir, optim, params['loss_fn_name'], batches=params['batches'], epochs = params['epochs'],
                  print_summary=False, splits=params['number_splits'], random_state=42)
    else:
        raise Exception("Please provide a valid cross-validation method")
        
param_dict = pd.DataFrame(list(params.items()), columns=['column1', 'column2'])
param_dict.to_csv(checkpoint_dir+'\\params.csv')

#%% Iteratively and combinatorially training models with validation specified in params dictionary
for optim, lossf, dropout  in product(params['opt_name'], params['loss_fn_name'], params['dropout']):

    run_string = (str(params['training_sample'])+"_"+ str(params['patch_size'])+"_"+ str(params['filters'])+"_"+str(dropout)+
                  "_"+str(params['batches'])+"_"+str(params['num layers'])+"_"+str(optim)+"_"+str(lossf))
    run_checkpoint_dir = os.path.join(checkpoint_dir, run_string)

    # Train model
    if crossval_method == "lpo":

        lpo_train(input_trainingPt, target_trainingPt, input_shape, num_classes, params['filters'], dropout,
                  params['num layers'], run_checkpoint_dir, optim, lossf, batches=params['batches'],
                  epochs = params['epochs'], print_summary=False, p_out=params['p-out'])

    elif crossval_method == "sss":

        sss_train(input_trainingPt, target_trainingPt, input_shape, num_classes, params['filters'], params['dropout'],
                  params['num layers'], run_checkpoint_dir, optim, lossf, path_string_list = patient_paths,
                  training_length=params['training_sample'],  batches=params['batches'], epochs = params['epochs'], print_summary=False,
                  test_size=params['test_size'], n_splits=params['number_splits'], random_state=42)

    elif crossval_method == "kfold":

        kfold_train(input_trainingPt, target_trainingPt, input_shape, num_classes, params['filters'], params['dropout'],
                  params['num layers'], run_checkpoint_dir, optim, lossf, batches=params['batches'], epochs = params['epochs'],
                  print_summary=False, splits=params['number_splits'], random_state=42)
    else:
        raise Exception("Please provide a valid cross-validation method")
        
param_dict = pd.DataFrame(list(params.items()), columns=['column1', 'column2'])
param_dict.to_csv(checkpoint_dir+'\\params.csv')



#%%
print(len(input_trainingPt), len(target_trainingPt[0]),len(target_trainingPt[1]),len(target_trainingPt[2]))
#transform(input_trainingPt,3)

for i in range(len(input_trainingPt)):
    print(i)

#%%
classes_dist = []

for i in range(3):
        classes_dist.append(int(patient_paths[i][-9]))
#%%
from tensorflow.keras.optimizers import Adam, RMSprop #Adagrad, SGD,
from IRCAD.losses import (loss_recall_favoured, dice_coef_loss, combo_loss) #focal_tversky_loss, categorical_cross_entropy, jaccard_distance
from IRCAD.utils import (gpu_check, pixel_count, predict_masks, binary_plot_image_and_masks)
#%% Create model
model = custom_unet(
    input_shape=(128,128,1),#input_shape,
    num_classes=3,
    filters=32,
    use_batch_norm=False,
    dropout=0.3,
    dropout_change_per_layer=0.0,
    num_layers=4,
    output_activation="softmax") #mutually exclusive outcomes = softmax

opt = Adam()
loss_fn = loss_recall_favoured
Mets = metric_tf_initialise(len(params['organs'])+1)

#Compile model
model.compile(
    optimizer=opt,
    loss=loss_fn,
    metrics=Mets,
)

#model_loc = (r"D:\PhD\Article2_AI Semantic Segmentation\Data\BatchNorm-T_LR-0.001_RFw-Norm_RFb-2-0.5-0.5_Aug-N_CCE-0.2\10_128_32_0.3_32_4_Adam_Recall Favoured\run_9\model-[0 1 2 3 4 5 6 7 8][9].h5")
model_loc = (r"D:\PhD\Article2_AI Semantic Segmentation\Data\BatchNorm-F_LR-0.001_RFw-Norm_RFb-2-0.5-0.5_Aug-N\10_128_32_0.3_32_4_Adam_Recall Favoured\run_9\model-[0 1 2 3 4 5 6 7 8][9].h5")

model.load_weights(model_loc)
#%%
validation_sample = input_validationPt[1]
validation_masks = target_validationPt[1]
pred_full_masks_class = predict_masks(validation_sample, model=model,
                                      size=params['patch_size'], stride=params['patch_size'])
#%%
for i in range(len(pred_full_masks_class)):
    binary_plot_image_and_masks(f'Patient Image {i}', validation_sample[i], validation_masks[i],
                          pred_full_masks_class[i], params['organs'])
