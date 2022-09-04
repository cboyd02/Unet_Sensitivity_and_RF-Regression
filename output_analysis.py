# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:40:02 2021
Last modified 1/12/2021
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from itertools import product
pd.set_option('display.max_rows',100)
pd.set_option('min_rows', None)
#%%
def write_csv_df(path, filename, df):
    # Give the filename you wish to save the file to
    pathfile = os.path.normpath(os.path.join(path,filename))

    # Use this function to search for any files which match your filename
    files_present = os.path.isfile(pathfile)
    # if no matching files, write to csv, if there are matching files, print statement
    if not files_present:
        df.to_csv(pathfile, sep=';')
    else:
        overwrite = input("WARNING: " + pathfile + " already exists! Do you want to overwrite <y/n>? \n ")
        if overwrite == 'y':
            df.to_csv(pathfile, sep=';')
        elif overwrite == 'n':
            new_filename = input("Type new filename: \n ")
            write_csv_df(path,new_filename,df)
        else:
            print("Not a valid input. Data is NOT saved!\n")

#%% FILTER CSV TO GET LAST EPOCH RESULTS ONLY AND BEST RESULTS ONLY, SAVING BOTH TO CSV
#path = "D:\\PhD\\Programming Practice\\Keras\\Sensitivity\\Results\\forprocessing"
path = "D:\\PhD\\Programming Practice\\Keras\\Sensitivity\\Sensitivity\\checkpoints"
#path = r"D:\PhD\Programming Practice\Keras\Sensitivity\Results\20220401_RecallFavoured-Investigation\1-2-2-NoAug"
df = []
df = pd.DataFrame()
for root, folders, files in os.walk(path): #should use list here rather than iteratively build df
    for file in  files:
        if file.startswith('params'):
            pass
        elif file.endswith('.csv'):
            filepath = os.path.join(root, file)
            print(filepath)
            df_combined_single = pd.read_csv(filepath)
            df = pd.concat((df, df_combined_single))
            df_testing = df.drop(["val_loss", "val_accuracy", "val_categorical_accuracy","val_JI_mean", "val_F1_mean",
                                  "val_JI_0", "val_JI_1", "val_JI_2", 
                                  "val_F1_0","val_F1_1", "val_F1_2",
                                  "val_Recall_0", "val_Recall_1", "val_Recall_2",
                                  "val_Precision_0", "val_Precision_1","val_Precision_2",
                                  ], axis=1)#, "lr"], axis=1)
            df_validation = df.drop(["loss", "accuracy", "categorical_accuracy","JI_mean", "F1_mean",
                                     "JI_0", "JI_1", "JI_2",
                                     "F1_0", "F1_1", "F1_2",
                                     "Recall_0", "Recall_1", "Recall_2",
                                     "Precision_0", "Precision_1","Precision_2",
                                     ], axis=1)#, "lr"], axis=1)
            print(df.shape, df_validation.shape)
        else:
            pass
labels = ["Optimiser", "Loss Fn",'leaveout', "Batches","Patch Size", "Test Sample Size",  "U-net Depth", "Epoch No."]
#labels = ["Optimiser", "Loss Fn",'splits', "Batches","Patch Size", "Test Sample Size",  "U-net Depth", "Epoch No."]

#Write full results to csv
df.to_csv(os.path.join(path, "CombinedResults.csv"), index=False, header=True)

#Group testing dataset by independent variables and calculate mean and std dev for each epoch
df_testing_error = df_testing.groupby(by=labels, as_index=False).std()
df_testing = df_testing.groupby(by=labels, as_index=False).mean()
df_testing_error.to_csv(os.path.join(path, "CombinedTestingError.csv"), index=False, header=True)
df_testing.to_csv(os.path.join(path, "CombinedTestingResults.csv"), index=False, header=True)

#Group validation dataset by independent variables and calculate mean and std dev for each epoch
df_validation_error = df_validation.groupby(by=labels, as_index=False).std()
df_validation = df_validation.groupby(by=labels, as_index=False).mean()
df_validation_error.to_csv(os.path.join(path, "CombinedValidationError.csv"), index=False, header=True)
df_validation.to_csv(os.path.join(path, "CombinedValidationResults.csv"), index=False, header=True)

#%%
path = "D:\\PhD\\Programming Practice\\Keras\\Sensitivity\\Sensitivity\\checkpoints"

df_raw = pd.read_csv(os.path.join(path, 'CombinedResults.csv'))

#%% Group by optimiser (hue) and loss function (line type) - TESTING RESULTS
plt.figure(figsize=(30,30))
plt.title("TRAINING Results with optimiser (hue) and loss (line type)", fontsize=40)

y_s=["loss", "accuracy", "categorical_accuracy","JI_mean", "F1_mean",
                         "JI_0", "JI_1", "JI_2",
                         "F1_0", "F1_1", "F1_2",
                         "Recall_0", "Recall_1", "Recall_2",
                         "Precision_0", "Precision_1","Precision_2"]

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
    for j in range(len(g.get_lines())):
        if len(g.get_lines()[j].get_data()[1]) == 0: pass
        else: print(g.get_lines()[j].get_data()[1][-1])

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

#%% Group by optimiser (hue) and loss function (line type) - VALIDATION RESULTS
plt.figure(figsize=(30,30))
plt.title("VALIDATION Results with optimiser (hue) and loss (line type)", fontsize=40)

y_s=["val_loss", "val_accuracy", "val_categorical_accuracy","val_JI_mean", "val_F1_mean",
                      "val_JI_0", "val_JI_1", "val_JI_2",
                      "val_F1_0","val_F1_1", "val_F1_2",
                      "val_Recall_0", "val_Recall_1", "val_Recall_2",
                      "val_Precision_0", "val_Precision_1","val_Precision_2"]
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
    for j in range(len(g.get_lines())):
        if len(g.get_lines()[j].get_data()[1]) == 0: pass
        else: print(g.get_lines()[j].get_data()[1][-1])
        
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)


#%% Group by optimiser, loss function and dropout
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
          'loss_fn_name': ["Recall Favoured", "Combo Loss"], 
          'dropout': [0.1, 0.3, 0.5],
          }

dim = int(np.ceil(np.sqrt(len(y_s))))
for optim, lossf, dropout  in product(params['opt_name'], params['loss_fn_name'], params['dropout']):
    i=1
    dftest = df_raw[(df_raw['Optimiser']==optim)& (df_raw['Loss Fn']==lossf)&(df_raw['Dropout']==dropout)]#df_raw.loc[lambda df_raw: df_raw['Optimiser']=='Adam']
    plt.figure(figsize=(30,30))

    for key, value in y_s.items():
        plt.subplot(dim,dim,i)
        melted = pd.melt(dftest, id_vars=['Epoch No.'], value_vars=y_s[key])
        g=sns.lineplot(data=melted, x='Epoch No.', y='value', hue='variable')
        g.set(ylim=(0,1), title=key)
        for j in range(len(g.get_lines())):
            if len(g.get_lines()[j].get_data()[1]) == 0: pass
            else: print(g.get_lines()[j].get_data()[1][-1])
        i+=1

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.suptitle(optim + " " + lossf + " " + str(dropout), fontsize=30)
    plt.show()
    
#%% Random Forest Analysis of output data - Adapted from: https://alphascientist.com/hyperparameter_optimization_with_talos.html
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

collated_path = r"D:\PhD\Article2_AI Semantic Segmentation\Data"
df_collated = pd.read_csv(os.path.join(collated_path, 'Publication_CombinedResults.csv'))

df_collated['Optimiser']=df_collated['Optimiser'].astype('category').cat.codes
df_collated['Loss Fn']=df_collated['Loss Fn'].astype('category').cat.codes
df_collated['Batchnorm']=df_collated['Batchnorm'].astype('category').cat.codes
df_collated['Recall Weights']=df_collated['Recall Weights'].astype('category').cat.codes
df_collated['Recall Betas']=df_collated['Recall Betas'].astype('category').cat.codes
df_collated['Augmentation']=df_collated['Augmentation'].astype('category').cat.codes


#%%
X = df_collated[['Optimiser', 'Loss Fn', 'Dropout', 'Batchnorm','lr','Recall Weights','Recall Betas','Augmentation']]
scaler = MinMaxScaler()

def rel_HP_imp(fitparam, cmap, titlestring, depth=5, estimators=1000, dp=3):
    y = scaler.fit_transform(df_collated[[fitparam]])
    reg = RandomForestRegressor(max_depth=depth,n_estimators=estimators)
    reg.fit(X,y.ravel())
    rel_imp = pd.Series(reg.feature_importances_,index=X.columns).sort_values(ascending=False)
    rel_imp.rename({'lr':'Learning Rate'}, inplace=True)
    rel_imp.rename({'Loss Fn':'Loss Function'}, inplace=True)
    
    ax = sns.barplot(x=rel_imp.values.round(dp), y=rel_imp.index, palette=cmap)

    # Decorations
    plt.title('Hyperparameter importance (by '+titlestring+')', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set(xlabel="Relative feature importance")
    ax.bar_label(ax.containers[0])
    
    
    plt.xlim(0,1.0)
    plt.show()
    
#%%By validation accuracy
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
rel_HP_imp('val_accuracy', 'Validation Accuracy')
#%%By Mean F1 Score of Organs
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
rel_HP_imp('val_F1_orgmean', 'Mean Organ DSC')
#%%By Mean JI of Organs
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
rel_HP_imp('val_JI_orgmean', 'Mean Organ Jaccard Index')
#%%By Mean Precision of Organs
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
rel_HP_imp('val_Precision_orgmean', 'Mean Organ Precision')
#%%By Mean Recall of Organs
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
rel_HP_imp('val_Recall_orgmean', 'Mean Organ Recall')
#%%By Mean Recall 
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
rel_HP_imp('val_Recall_mean', 'Mean Recall inc. Background')
#%%By Mean Precision
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
rel_HP_imp('val_Precision_mean', 'Mean Precision inc. Background')