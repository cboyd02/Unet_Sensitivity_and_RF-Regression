import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from itertools import product
pd.set_option('display.max_rows',100)
pd.set_option('min_rows', None)

#%% FILTER CSV TO GET LAST EPOCH RESULTS ONLY AND BEST RESULTS ONLY, SAVING BOTH TO CSV
path = r"[PLACEHOLDER PATH] - Map to checkpoints directory generated during training"
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

#%% Import combined results csv
df_raw = pd.read_csv(os.path.join(path, 'CombinedResults.csv'))

#%% Generate all figues possible, grouped by optimiser (hue) and loss function (line type) - TESTING RESULTS
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

#%% Generate all figues possible, grouped by optimiser (hue) and loss function (line type) - VALIDATION RESULTS
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

#%% Generate all figues possible, grouped by optimiser, loss function and dropout
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
    dftest = df_raw[(df_raw['Optimiser']==optim)& (df_raw['Loss Fn']==lossf)&(df_raw['Dropout']==dropout)]
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

#Navigate to path and import results into dataframe
collated_path = r"[PLACEHOLDER PATH] - Map to user made 'Data' directory"
df_collated = pd.read_csv(os.path.join(collated_path, 'Publication_CombinedResults.csv'))

#Assign numbers to non-numerical values so RF regression can be performed
df_collated['Optimiser']=df_collated['Optimiser'].astype('category').cat.codes
df_collated['Loss Fn']=df_collated['Loss Fn'].astype('category').cat.codes
df_collated['Batchnorm']=df_collated['Batchnorm'].astype('category').cat.codes
df_collated['Recall Weights']=df_collated['Recall Weights'].astype('category').cat.codes
df_collated['Recall Betas']=df_collated['Recall Betas'].astype('category').cat.codes
df_collated['Augmentation']=df_collated['Augmentation'].astype('category').cat.codes

#Generate custom colourmap
series_cmap = {'Optimiser':("#9400D3"), 'Loss Function':("#FF1493"), 'Dropout':("#4B0082"), 'Batchnorm':("#0000FF"),
                   'Learning Rate':("#00FF00"),'Recall Weights':("#FFFF00"),'Recall Betas':("#FF7F00"),
                   'Augmentation':("#FF0000")}
#%%
X = df_collated[['Optimiser', 'Loss Fn', 'Dropout', 'Batchnorm','lr','Recall Weights','Recall Betas','Augmentation']]
scaler = MinMaxScaler()

#Use SKLearn RFRegressor on data to compare relative importance of X
def rel_HP_imp(fitparam, cmap, titlestring, depth=10, estimators=1000, dp=3):
    y = scaler.fit_transform(df_collated[[fitparam]])
    reg = RandomForestRegressor(max_depth=depth, n_estimators=estimators)
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
    
#%%Plot RF regression of relative hyperparameter importance - By validation accuracy
rel_HP_imp('val_accuracy', series_cmap, 'Validation Accuracy')
#%%Plot RF regression of relative hyperparameter importance - By Mean F1 Score of Organs
rel_HP_imp('val_F1_orgmean',series_cmap, 'Mean Organ DSC')
#%%Plot RF regression of relative hyperparameter importance - By Mean JI of Organs
rel_HP_imp('val_JI_orgmean', series_cmap, 'Mean Organ Jaccard Index')
#%%Plot RF regression of relative hyperparameter importance - By Mean Precision of Organs
rel_HP_imp('val_Precision_orgmean', series_cmap, 'Mean Organ Precision')
#%%Plot RF regression of relative hyperparameter importance - By Mean Recall of Organs
rel_HP_imp('val_Recall_orgmean', series_cmap, 'Mean Organ Recall')
#%%Plot RF regression of relative hyperparameter importance - By Mean Recall 
rel_HP_imp('val_Recall_mean', series_cmap, 'Mean Recall inc. Background')
#%%Plot RF regression of relative hyperparameter importance - By Mean Precision
rel_HP_imp('val_Precision_mean', series_cmap, 'Mean Precision inc. Background')