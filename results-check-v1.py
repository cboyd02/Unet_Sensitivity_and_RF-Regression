#!/usr/bin/env python
# coding: utf-8
#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import product
pd.set_option('display.max_rows',100)
pd.set_option('min_rows', None)

#%%
path = r"D:\PhD\Programming Practice\Keras\Sensitivity\Sensitivity\checkpoints"
#path = r"D:\PhD\Programming Practice\Keras\Sensitivity\Results\20220401_RecallFavoured-Investigation\0.5-2-2-NoAug"
df_raw = pd.read_csv(os.path.join(path, 'CombinedResults.csv'))

#%%


df_raw.columns

#%%


# these are the columns that characterise each setting.
# FIXME: is 'Test Sample Size' a hyper-parameter that was varied?
#id_cols = ['Optimiser', 'Loss Fn', 'Batches', 'Patch Size', 'leaveout', 'Test Sample Size', 'U-net Depth']
id_cols = ['Model filters', 'Dropout', 'U-net Depth']
#id_cols = ['Optimiser', 'Loss Fn']

#%%
# average over all folds in each setting and epoch
df_xval = df_raw.drop(columns=['train','val','lr']).groupby(id_cols + ['Epoch No.'] + ['fold']).mean()

#%%
sns.lineplot(x='Epoch No.', y='loss', hue='Model filters', data=df_xval)

#%%
# average over all folds in each setting and epoch
df_xval = df_raw.drop(columns=['train','val','fold','lr']).groupby(id_cols + ['Epoch No.']).mean()

#%%


df_xval.groupby(id_cols).count()
# FIXME: why does the number of epochs vary across settings? (mostly 20, but some settings have 10 or 18 epochs)

#%%


# find the epoch with best validation loss (mimic model selection in each setting)
df_best = df_xval.reset_index()
df_best = df_best.loc[df_best.groupby(id_cols).val_loss.idxmin()]
df_best

#%%


# FIXME: why are there only 60 different settings? with all combinations of values for each of the hyperparameters this should be 2304 settings by my calculation (or 768 if test sample size is not a parameter that was varied).
# create a dataframe with NaN for each setting that is absent
df_full = df_best.set_index(id_cols)
full_idx = pd.MultiIndex.from_product(df_full.index.levels)
df_full.reindex(full_idx).reset_index()

#%%


# look for relationships among validation metrics
sns.pairplot(df_best[['val_loss', 'val_accuracy', 'val_categorical_accuracy',
                      'val_JI_mean', 'val_F1_mean','val_JI_0', 'val_JI_1',
                      'val_JI_2', 'val_F1_0', 'val_F1_1', 'val_F1_2','val_Recall_0',
                      'val_Recall_1', 'val_Recall_2', 'val_Precision_0','val_Precision_1', 'val_Precision_2']]);
# it seems that recall and F1 scores are poor for classes 1 and 2.

#%%

# look for relationships among hyperparameters and val_loss/val_dice_coef.
# this does not reveal much currently, due to the biased coverage of the hyperparameter scenarios.
sns.pairplot(df_best[['Model filters', 'Dropout', 'U-net Depth', 'val_loss', 'val_F1_mean']]);

#%%
print(df_raw.keys)
#%% Group by optimiser (hue) and loss function (line type)
plt.figure(figsize=(30,30))
plt.title("TRAINING Results with optimiser (hue) and loss (line type)", fontsize=40)

y_s=['loss','accuracy', 'JI_0', 'JI_1', 'JI_2','F1_0','F1_1',
     'F1_2', 'Recall_0', 'Recall_1', 'Recall_2', 'Precision_0', 'Precision_1',
     'Precision_2']

dim = int(np.ceil(np.sqrt(len(y_s))))

print(dim)

for i in range(len(y_s)):
    print('up to figure {}'.format(i))
    plt.subplot(dim,dim,i+1)
    if i == 0:
        g=sns.lineplot(data=df_raw, x='Epoch No.', y=y_s[i], hue='Optimiser', style="Loss Fn")
    else:
        g = sns.lineplot(data=df_raw, x='Epoch No.', y=y_s[i], hue='Optimiser', style="Loss Fn", legend=False)
    g.set(ylim=(0,1))

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

#%% Group by optimiser (hue) and loss function (line type)
plt.figure(figsize=(30,30))
plt.title("VALIDATION Results with optimiser (hue) and loss (line type)", fontsize=40)

y_s=['val_loss', 'val_accuracy', 'val_JI_0', 'val_JI_1', 'val_JI_2','val_F1_0','val_F1_1',
     'val_F1_2', 'val_Recall_0', 'val_Recall_1', 'val_Recall_2', 'val_Precision_0', 'val_Precision_1',
     'val_Precision_2']
dim = int(np.ceil(np.sqrt(len(y_s))))
print(dim)
for i in range(len(y_s)):
    print('up to figure {}'.format(i))
    plt.subplot(dim,dim,i+1)
    if i == 0:
        g=sns.lineplot(data=df_raw, x='Epoch No.', y=y_s[i], hue='Optimiser', style="Loss Fn", palette='prism')
    else:
        g = sns.lineplot(data=df_raw, x='Epoch No.', y=y_s[i], hue='Optimiser', style="Loss Fn", palette='prism', legend=False)
    g.set(ylim=(0,1))

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
#%% Group by class (hue)
plt.figure(figsize=(30,30))

y_s={'JI':['JI_0', 'JI_1','JI_2'],
     'F1':['F1_0', 'F1_1','F1_2'],
     'precision':['Precision_0', 'Precision_1','Precision_2'],
     'recall':['Recall_0', 'Recall_1', 'Recall_2'],
     'loss':['loss', 'val_loss'],
     'accuracy':['accuracy','val_accuracy'],
     'Validation JI':['val_JI_0', 'val_JI_1','val_JI_2'],
     'Validation F1':['val_F1_0', 'val_F1_1','val_F1_2'],
     'Validation precision':['val_Precision_0', 'val_Precision_1','val_Precision_2'],
     'Validation recall':['val_Recall_0', 'val_Recall_1', 'val_Recall_2']}

params = {'opt_name': ["Adam", "RMSprop"], #"Adagrad", "SGD",
          'loss_fn_name': ["Recall Favoured", "CCE Weighted DSC", "Focal Tversky Loss"], #"Dice Loss",
          }

dim = int(np.ceil(np.sqrt(len(y_s))))
for optim, lossf  in product(params['opt_name'], params['loss_fn_name']):
    i=1
    dftest = df_raw[(df_raw['Optimiser']==optim)& (df_raw['Loss Fn']==lossf)]#df_raw.loc[lambda df_raw: df_raw['Optimiser']=='Adam']
    plt.figure(figsize=(30,30))

    for key, value in y_s.items():
        plt.subplot(dim,dim,i)
        melted = pd.melt(dftest, id_vars=['Epoch No.'], value_vars=y_s[key])
        g=sns.lineplot(data=melted, x='Epoch No.', y='value', hue='variable')
        g.set(ylim=(0,1), title=key)
        i+=1

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.suptitle(optim + " " + lossf)
    plt.show()

#%% Group by class (hue)
plt.figure(figsize=(30,30))

y_s={'JI':['JI_0', 'JI_1','JI_2'],
     'F1':['F1_0', 'F1_1','F1_2'],
     'precision':['Precision_0', 'Precision_1','Precision_2'],
     'recall':['Recall_0', 'Recall_1', 'Recall_2'],
     'loss':['loss', 'val_loss']}

dim = int(np.ceil(np.sqrt(len(y_s))))
print(dim)
for key, value in y_s.items():
    print(key)
    for i in range(len(y_s[key])):
        print('up to figure {}'.format(i))
        plt.subplot(dim,dim,i+1)
        g=sns.lineplot(data=df_raw, x='Epoch No.', y=y_s[key][i])

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)