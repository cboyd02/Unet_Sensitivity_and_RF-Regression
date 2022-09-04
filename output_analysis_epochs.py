# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:40:02 2021
Last modified 2/9/22
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
pd.set_option('display.max_rows',100)
pd.set_option('min_rows', None)

#%% FILTER TO GET LAST EPOCH RESULTS ONLY AND BEST RESULTS ONLY
path = r"D:\PhD\Article2_AI Semantic Segmentation\Data\_Figures\EpochTesting"
df = []
df = pd.DataFrame()
file = 'EpochTesting.xlsx'
filepath = os.path.join(path, file)
print(filepath)
df_combined_single = pd.read_excel(filepath, sheet_name='Sheet2')
df = pd.concat((df, df_combined_single))
print(df.shape)
       
#Write full results to csv
df.to_csv(os.path.join(path, "CombinedResults.csv"), index=False, header=True)

#%% Group by total epoch (hue) and optimiser (line type) - TESTING RESULTS
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
        g=sns.barplot(data=df, x='MaxEpoch', y=y_s[i], palette='OrRd', ci=95)
    else:
        g = sns.barplot(data=df, x='MaxEpoch', y=y_s[i], palette='OrRd', ci=95)
    g.set(ylim=(0,1))

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

#%% Group by total epoch (hue) and optimiser (line type) - VALIDATION RESULTS
plt.figure(figsize=(30,30))
plt.title("VALIDATION Results with number of epoch (hue) and optimiser (line type)", fontsize=40)

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
        g=sns.barplot(data=df, x='MaxEpoch', y=y_s[i], palette='GnBu')
    else:
        g = sns.barplot(data=df, x='MaxEpoch', y=y_s[i], palette='GnBu')
    g.set(ylim=(0,1))
        
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

#%%

g=sns.barplot(data=df, x='MaxEpoch', y="loss", palette='OrRd', ci=95, capsize=0.3)

g.set(ylabel="Testing Loss", ylim=(0,0.25))

#%%
g=sns.barplot(data=df, x='MaxEpoch', y="val_loss", palette='GnBu', ci=95, capsize=0.3)

g.set(ylabel="Validation Loss", ylim=(0,0.25))

