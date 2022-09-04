# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:40:02 2021
Last modified 1/12/2021
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

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
#path = r"D:\PhD\Programming Practice\Keras\Sensitivity\Results\20220305_LOOCV_Augment_10pat"
df = []
df = pd.DataFrame()
for root, folders, files in os.walk(path): #should use list here rather than iteratively build df
    for file in  files:
        if file.endswith('.csv'):
            filepath = os.path.join(root, file)
            print(filepath)
            df_combined_single = pd.read_csv(filepath)
            df = pd.concat((df, df_combined_single))
            df_validation = df.drop(["loss", "accuracy", "categorical_accuracy",
                                              "JI_mean", "F1_mean", "JI_0", "JI_1", "JI_2", "F1_0", "F1_1", "F1_2",
                                              "Recall_0", "Recall_1", "Recall_2", "Precision_0", "Precision_1",
                                              "Precision_2", "lr"], axis=1)
            print(df.shape, df_validation.shape)
        else:
            pass
labels = ["Optimiser", "Loss Fn",'leaveout', "Batches","Patch Size", "Test Sample Size",  "U-net Depth", "Epoch No."]
#labels = ["Optimiser", "Loss Fn",'splits', "Batches","Patch Size", "Test Sample Size",  "U-net Depth", "Epoch No."]

df_validation_error = df_validation.groupby(by=labels, as_index=False).std()
df_validation = df_validation.groupby(by=labels, as_index=False).mean()

df.to_csv(os.path.join(path, "CombinedResults.csv"), index=False, header=True)
df_validation_error.to_csv(os.path.join(path, "CombinedValidationError.csv"), index=False, header=True)
df_validation.to_csv(os.path.join(path, "CombinedValidationResults.csv"), index=False, header=True)

#%%
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Version
print(mpl.__version__)  #> 3.0.0
print(sns.__version__)  #> 0.9.0

#%%
in_path = "D:\\PhD\\Programming Practice\\Keras\\Sensitivity\\Results\\"
in_file = os.path.join(path, "CombinedValidationResults.csv")
in_df = pd.read_csv(in_file)
# Convert string parameters to category codes
in_df['Optimiser']=in_df['Optimiser'].astype('category').cat.codes
in_df['Loss Fn']=in_df['Loss Fn'].astype('category').cat.codes
in_df = in_df.drop(["fold"], axis=1)
# Plot
plt.figure(figsize=(12,10), dpi= 300)
corr = df_validation.corr(method='kendall')

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr.abs(), mask=mask, xticklabels=df_validation.corr().columns, yticklabels=df_validation.corr().columns,
            cmap='RdYlGn', center=0, annot=True, annot_kws={"fontsize":8})

# Decorations
plt.title('Correlogram', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#%%
sns.jointplot(x=in_df['val_loss'], y=in_df['Batches'], kind='reg')
plt.show()
#%%
sns.jointplot(x=in_df['val_loss'], y=in_df['val_Recall_0'], kind='reg')
plt.show()

#%%
sns.jointplot(x=in_df['val_loss'], y=in_df['val_Precision_1'], kind='reg')
plt.show()
#%%
sns.jointplot(x=in_df['val_loss'], y=in_df['val_JI_0'], ylim=(0,1), kind='reg')
plt.show()
#%%
sns.jointplot(x=in_df['val_accuracy'], y=in_df['val_JI_mean'],xlim=[0,1],ylim=[0,1],kind='reg')
plt.show()
#%%
sns.scatterplot(x=in_df['val_loss'], y=in_df['val_Precision_1'])
plt.show()