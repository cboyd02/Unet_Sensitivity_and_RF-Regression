# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:15:21 2021

@author: chris

Adapted from Dr Wolfgang Mayer

There are libraries that carry out such augmentations, see eg. https://albumentations.ai/docs/getting_started/mask_augmentation/ 
and https://albumentations.ai/docs/examples/example_kaggle_salt/
"""
#%%
import os, sys
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)

#%%
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import LeavePOut, StratifiedShuffleSplit, KFold
from utils import try_free_memory
from random import shuffle
from models.custom_unet import custom_unet
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
#from tensorflow_addons.optimizers import AdamW
from IRCAD.losses import (jaccard_distance, loss_recall_favoured,
                          dice_coef_loss, focal_tversky_loss, cce_loss, combo_loss)
from IRCAD.metrics_tf import metric_tf_initialise
import time
#%%
def train_val_model(x_train, y_train, x_val, y_val, name, input_shape, num_classes, filters, dropout, num_layers,
                    checkpoint_dir, opt_name, loss_fn_name, run_no, batches=64, epochs=20,  print_summary=False, ClassWeights=[0.002,0.597,0.401]):
    
    model = custom_unet(
      input_shape=input_shape,
      num_classes=num_classes,
      filters=filters,
      use_batch_norm=True,
      dropout=dropout,
      dropout_change_per_layer=0.0,
      num_layers=num_layers,
      output_activation="softmax")
            
    #Re-call model parameters to reset weights
    #Horrific brute force - Cleanup
    lr = 0.010 #0.001 0.005 0.01
    if opt_name == "Adam": 
        opt = Adam(learning_rate=lr)
    elif opt_name == "Adagrad":
        opt = Adagrad()
    # elif opt_name == "AdamW": #weighted Adam not in TF current and tf_addons crashed CudNN compile 8/12/21
    #     opt = AdamW()
    elif opt_name == "SGD":
        opt = SGD()
    elif opt_name == "RMSprop":
        opt = RMSprop(learning_rate=lr)
    else:
        ValueError
    
    if loss_fn_name == "Recall Favoured":
        loss_fn = loss_recall_favoured(num_classes, ClassWeights = [0.5,0.5,0.5], beta = 0.5, BG_weight = 2)
    elif loss_fn_name == "Jaccard Distance":
        loss_fn = jaccard_distance()
    elif loss_fn_name == "Dice Loss":
        loss_fn = dice_coef_loss()
    elif loss_fn_name == "Focal Tversky Loss":
        loss_fn = focal_tversky_loss(smooth=1, alpha=0.7, gamma=1.33)
    elif loss_fn_name == "Combo Loss":
        loss_fn = combo_loss(cce_weight = 0.4) #0.05 0.2 0.4
    # elif loss_fn_name == "CCE":
    #     loss_fn = cce_loss
    else:
        ValueError

    Mets = metric_tf_initialise(num_classes)
    
    #Compile model
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=Mets,
    )

    if print_summary:
        print(model.summary())

    # checkpoint the best model
    checkpoint_run = os.path.join(checkpoint_dir, str("run_")+str(run_no))

    pathlib.Path(checkpoint_run).mkdir(parents=True, exist_ok=True)
       
    checkpoint_file = f'{checkpoint_run}/model-{name}.h5'
    
    callback_checkpoint = ModelCheckpoint(
      filepath=checkpoint_file,
      verbose=1, 
      monitor='val_loss', 
      save_best_only=True,
      save_weights_only=True
    )

    #early_stopping = EarlyStopping(patience=10)

    lr_decay_auto = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1)
    tensorboard_callback = TensorBoard(log_dir= f'{checkpoint_run}/logs', histogram_freq=1,
                                       update_freq='epoch') #Navigate to run_N directory in Conda and execute in console: tensorboard --logdir logs
    #print('shape and type of x_train is: {} {} y_train is: {} {}'.format(x_train.shape, type(x_train), y_train.shape, type(y_train)))
    
    history = model.fit(x_train, y_train,
        batch_size=batches,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[callback_checkpoint, tensorboard_callback]# lr_decay_auto, early_stopping] - Removed early stop for reproducibility
    )

    return history

def kfold_train(images, masks,  input_shape, num_classes, filters, dropout, num_layers,
              checkpoint_dir, opt_name, loss_fn_name, batches=64, epochs=20, print_summary=False, splits=10, random_state=0):
    
    history = []
    kf = KFold(n_splits=splits)
    kf.get_n_splits
    i=0
    for patients_train, patients_val in kf.split(images):
        print('Training: {}, Validation: {}'.format(patients_train, patients_val))
        
        name = str(patients_train) + str(patients_val)
        
        # try free-up memory
        try_free_memory()

        # make data sets
        x_train, x_val = np.concatenate(images[patients_train]), np.concatenate(images[patients_val]),
        y_train, y_val = np.concatenate(masks[patients_train]), np.concatenate(masks[patients_val])
        
       
        # train
        history_fold = train_val_model(x_train, y_train, x_val, y_val, name, input_shape, num_classes, filters, dropout, num_layers,
                                       checkpoint_dir, opt_name, loss_fn_name, run_no=i, batches=batches, epochs=epochs, print_summary= False)#(fold==0))
        
        title = str(opt_name)+" "+str(loss_fn_name)+" "+str(i)
        plt.plot(history_fold.history['loss'], label='Training Loss', color='b', linestyle='dotted')
        plt.plot(history_fold.history['accuracy'], label='Training Accuracy', color='g',linestyle='dotted')
        plt.plot(history_fold.history['F1_mean'], label='Training Dice Mean', color='r', linestyle='dotted')
        plt.plot(history_fold.history['JI_mean'], label='Training Jaccard Mean', color='k', linestyle='dotted')
        plt.plot(history_fold.history['val_loss'], color='b', label='Validation Loss')
        plt.plot(history_fold.history['val_accuracy'], color='g', label='Validation Accuracy')
        plt.plot(history_fold.history['val_F1_mean'], color='r', label='Validation Dice Mean')
        plt.plot(history_fold.history['val_JI_mean'], color='k', label='Validation Jaccard Mean')
        plt.legend()
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylim(0,1.0)
        plt.show()
        
        # capture statistics
        df = pd.DataFrame.from_dict(history_fold.history).assign(train=str(patients_train),
                                                                 val=str(patients_val))
        df.insert(1, "Run No.", i)
        df.insert(2,"Optimiser", opt_name)
        df.insert(3, "Loss Fn", loss_fn_name)
        df.insert(4,"Batches", batches)
        df.insert(5, "splits", splits)
        df.insert(6, "Model filters", filters)
        df.insert(7, "Dropout", dropout)
        df.insert(8, "Patch Size", input_shape[0])
        df.insert(9, "Test Sample Size", len(images))
        df.insert(10, 'U-net Depth', num_layers)
        df.index.name = "Epoch No."

        history.append(df)
        i+=1
        del history_fold
        del df

    history = pd.concat(history)
        
        
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    history.to_csv(f'{checkpoint_dir}/'+str(timestr)+'_history.csv')
    return history


def lpo_train(images, masks,  input_shape, num_classes, filters, dropout, num_layers,
              checkpoint_dir, opt_name, loss_fn_name, batches=64, epochs=20, print_summary=False, p_out=1):
    lpo = LeavePOut(p=p_out)
    history = []
    i=0 # to label figures with run number
        
    for fold, (patients_train, patients_val) in enumerate(lpo.split(images)):
        print('Fold: {}, Training: {}, Validation: {}'.format(fold, patients_train, patients_val))
        name = str(patients_train) + str(patients_val)
        
        # try free-up memory
        try_free_memory()
        # make data sets
        x_train, x_val = np.concatenate(images[patients_train]), np.concatenate(images[patients_val]),
        y_train, y_val = np.concatenate(masks[patients_train]), np.concatenate(masks[patients_val])
       
        # train
        history_fold = train_val_model(x_train, y_train, x_val, y_val, name, input_shape, num_classes, filters, dropout, num_layers,
                                       checkpoint_dir, opt_name, loss_fn_name,  run_no=i, batches=batches, epochs=epochs, print_summary=(fold==0))
        title = str(opt_name)+" "+str(loss_fn_name)+" "+str(i)
        plt.plot(history_fold.history['loss'], label='Training Loss', color='b', linestyle='dotted')
        plt.plot(history_fold.history['accuracy'], label='Training Accuracy', color='g',linestyle='dotted')
        plt.plot(history_fold.history['F1_mean'], label='Training Dice Mean', color='r', linestyle='dotted')
        plt.plot(history_fold.history['JI_mean'], label='Training Jaccard Mean', color='k', linestyle='dotted')
        plt.plot(history_fold.history['val_loss'], color='b', label='Validation Loss')
        plt.plot(history_fold.history['val_accuracy'], color='g', label='Validation Accuracy')
        plt.plot(history_fold.history['val_F1_mean'], color='r', label='Validation Dice Mean')
        plt.plot(history_fold.history['val_JI_mean'], color='k', label='Validation Jaccard Mean')
        plt.legend()
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylim(0,1.0)
        plt.show()
        
        # capture statistics
        df = pd.DataFrame.from_dict(history_fold.history).assign(fold=fold, train=str(patients_train),
                                                                 val=str(patients_val))
        df.insert(1, "Run No.", i)
        df.insert(2,"Optimiser", opt_name)
        df.insert(3, "Loss Fn", loss_fn_name)
        df.insert(4,"Batches", batches)
        df.insert(5, "leaveout", p_out)
        df.insert(6, "Model filters", filters)
        df.insert(7, "Dropout", dropout)
        df.insert(8, "Patch Size", input_shape[0])
        df.insert(9, "Test Sample Size", len(images))
        df.insert(10, 'U-net Depth', num_layers)
        df.index.name = "Epoch No."

        history.append(df)
        i+=1
        del history_fold
        del df

    history = pd.concat(history)
        
        
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    history.to_csv(f'{checkpoint_dir}/'+str(timestr)+'_history.csv')
    return history

def sss_train(images, masks, input_shape, num_classes,  filters, dropout, num_layers, checkpoint_dir, opt_name, loss_fn_name, path_string_list,
              training_length, batches=64, epochs=20, print_summary=False, test_size=0.2, n_splits=10, random_state=0):
    
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    history = []

    classes_dist = []

    for i in range(training_length):
        classes_dist.append(int(path_string_list[i][-9]))

    classes = np.array(classes_dist)

    sss.get_n_splits(images, num_classes)
    i=0 
    for fold, (patients_train, patients_val) in enumerate(sss.split(images,num_classes)):
        print('Fold: {}, Training: {}, Validation: {}'.format(fold, patients_train, patients_val))
        name = str(patients_train) + str(patients_val)
        
        # try free-up memory
        try_free_memory()

        # make data sets
        x_train, x_val = np.concatenate(images[patients_train]), np.concatenate(images[patients_val]),
        y_train, y_val = np.concatenate(masks[patients_train]), np.concatenate(masks[patients_val])
        
       
        # train
        history_fold = train_val_model(x_train, y_train, x_val, y_val, name, input_shape, num_classes, filters, dropout, num_layers,
                                       checkpoint_dir, opt_name, loss_fn_name,  run_no=i, batches=batches, epochs=epochs, print_summary=(fold==0))

        title = str(opt_name)+" "+str(loss_fn_name)+" "+str(i)
        plt.plot(history_fold.history['loss'], label='Training Loss', color='b', linestyle='dotted')
        plt.plot(history_fold.history['accuracy'], label='Training Accuracy', color='g',linestyle='dotted')
        plt.plot(history_fold.history['F1_mean'], label='Training Dice Mean', color='r', linestyle='dotted')
        plt.plot(history_fold.history['JI_mean'], label='Training Jaccard Mean', color='k', linestyle='dotted')
        plt.plot(history_fold.history['val_loss'], color='b', label='Validation Loss')
        plt.plot(history_fold.history['val_accuracy'], color='g', label='Validation Accuracy')
        plt.plot(history_fold.history['val_F1_mean'], color='r', label='Validation Dice Mean')
        plt.plot(history_fold.history['val_JI_mean'], color='k', label='Validation Jaccard Mean')
        plt.legend()
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylim(0,1.0)
        plt.show()
        
        # capture statistics
        df = pd.DataFrame.from_dict(history_fold.history).assign(fold=fold, train=str(patients_train),
                                                                 val=str(patients_val))
        df.insert(1, "Run No.", i)
        df.insert(2,"Optimiser", opt_name)
        df.insert(3, "Loss Fn", loss_fn_name)
        df.insert(4,"Batches", batches)
        df.insert(5, "splits", n_splits)
        df.insert(6, "Model filters", filters)
        df.insert(7, "Dropout", dropout)
        df.insert(8, "Patch Size", input_shape[0])
        df.insert(9, "Test Sample Size", len(images))
        df.insert(10, 'U-net Depth', num_layers)
        df.index.name = "Epoch No."

        history.append(df)
        
        i+=1
        del history_fold
        del df

    history = pd.concat(history)
        
        
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    history.to_csv(f'{checkpoint_dir}/'+str(timestr)+'_history.csv')
    return history