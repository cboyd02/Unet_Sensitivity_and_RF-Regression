"""
Code taking output "CombinedResults.csv" generated during model training and producing a wide range of figures.
Figures produce include correlograms, boxplots, facetgrids, and complex line plots (varying with colour, style and size)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
pd.set_option('display.max_rows',100)
pd.set_option('min_rows', None)

###################################
# Define default display parameters
###################################

large = 24; med = 16; small = 12
params = {'axes.titlesize': med,
          'legend.fontsize': med,
          'legend.title_fontsize': large,
          'figure.figsize': (15, 15),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white", {'legend.frameon':True})
my_palette = sns.color_palette(palette=["#7C2222", "#efcbcb" ], as_cmap=True)
# Version
#print(mpl.__version__)  #> 3.0.0
print(sns.__version__)  #> 0.9.0

#%%
in_path = r"[PLACEHOLDER PATH] - Point to folder containing output csv data"
in_file = os.path.join(in_path, "Publication_CombinedResults.csv")
in_df = pd.read_csv(in_file)

#Convert string parameters to category codes
in_df['train']=in_df['train'].astype('category').cat.codes
in_df['val']=in_df['val'].astype('category').cat.codes
in_df = in_df.drop(["fold"], axis=1)

testing_labels = ["loss", "accuracy", "categorical_accuracy", "JI_mean", "F1_mean", "JI_0", "JI_1", "JI_2",
                  "F1_0", "F1_1", "F1_2", "Recall_0", "Recall_1", "Recall_2", "Precision_0", "Precision_1", 
                  "Precision_2", "Batches", "leaveout", "Model filters", "Patch Size", "Test Sample Size", "U-net Depth"]
validation_labels = ["val_loss", "val_accuracy", "val_categorical_accuracy", "val_JI_mean", "val_F1_mean", "val_JI_0",
                     "val_JI_1", "val_JI_2", "val_F1_0", "val_F1_1", "val_F1_2", "val_Recall_0", "val_Recall_1", "val_Recall_2",
                     "val_Precision_0", "val_Precision_1", "val_Precision_2", "Batches", "leaveout", "Model filters", "Patch Size",
                     "Test Sample Size", "U-net Depth"]

df_validation = in_df.drop(labels=testing_labels, axis=1)
df_testing = in_df.drop(labels=validation_labels, axis=1)

df_heatmap = df_validation.copy(deep=True)

df_heatmap['Batchnorm']=df_heatmap['Batchnorm'].astype('category').cat.codes
df_heatmap['Recall Weights']=df_heatmap['Recall Weights'].astype('category').cat.codes
df_heatmap['Recall Betas']=df_heatmap['Recall Betas'].astype('category').cat.codes
df_heatmap['Augmentation']=df_heatmap['Augmentation'].astype('category').cat.codes


df_rf_test = df_testing.drop(df_testing[df_testing['Loss Fn']=="Combo Loss"].index) #Remove "combo loss"
df_rf_val = df_validation.drop(df_validation[df_validation['Loss Fn']=="Combo Loss"].index) #Remove "combo loss"

df_cl_test = df_testing.drop(df_testing[df_testing['Loss Fn']=="Recall Favoured"].index) #Remove "Recall Favoured"
df_cl_val = df_validation.drop(df_validation[df_validation['Loss Fn']=="Recall Favoured"].index) #Remove "Recall Favoured"

#%% Recall Favoured Correllogram
df_rf_heatmap = df_heatmap[['Loss Fn','Dropout', 'val_loss', 'val_accuracy', 'Batchnorm', 'lr', 'Augmentation',
                            'Recall Weights','Recall Betas']].copy()
df_rf_heatmap = df_rf_heatmap.drop(df_rf_heatmap[df_rf_heatmap['Loss Fn']=="Combo Loss"].index) #Remove "Combo Loss"
df_rf_heatmap.drop(labels=['Loss Fn'], axis=1, inplace=True)
corr = df_rf_heatmap.corr(method='spearman')

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr.abs(), mask=mask, xticklabels=df_rf_heatmap.corr().columns, yticklabels=df_rf_heatmap.corr().columns,
            cmap='RdBu_r', center=0, fmt='1.3f', linewidths=.5, annot=True, annot_kws={"fontsize":16})

# Decorations
plt.title('Recall Favoured Correlogram', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
#%% Combo Loss Correllogram
# Plot
plt.figure(figsize=(12,10), dpi= 300)

df_cl_heatmap = df_heatmap[['Loss Fn','Dropout', 'val_loss', 'val_accuracy', 'Batchnorm', 'lr', 'Augmentation',
                            'Combo Loss Weight']].copy()
df_cl_heatmap = df_cl_heatmap.drop(df_cl_heatmap[df_cl_heatmap['Loss Fn']=="Recall Favoured"].index) #Remove "Recall Favoured"
df_cl_heatmap.drop(labels=['Loss Fn'], axis=1, inplace=True)

corr = df_cl_heatmap.corr(method='spearman')

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr.abs(), mask=mask, xticklabels=df_cl_heatmap.corr().columns, yticklabels=df_cl_heatmap.corr().columns,
            cmap='RdBu_r', center=0, fmt='1.3f', linewidths=.5, annot=True, annot_kws={"fontsize":16})

# Decorations
plt.title('Combo Loss Correlogram', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#%%
multibox = sns.catplot(x='Recall Betas', y='val_F1_mean', hue='Augmentation', col='Recall Weights', data=df_rf_val, kind="box", showfliers=False)
multibox.fig.subplots_adjust(top=0.85)
multibox.fig.suptitle("Augmentation of recall favoured loss")
multibox.set_axis_labels(y_var='Mean Dice Similarity Coefficient')
plt.show
#%%
box = sns.boxplot(x='Optimiser', y='F1_mean', hue= 'Batchnorm', data=df_testing, showfliers=False)
plt.title("Training Dice Similarity Coefficient (DSC)")
plt.ylabel('Mean Dice Similarity Coefficient')
plt.ylim(0,1)
plt.show
#%%
box = sns.boxplot(x='Optimiser', y='val_F1_mean', hue= 'Batchnorm', data=df_validation, showfliers=False)
plt.title("Validation Dice Similarity Coefficient (DSC)")
plt.ylabel('Mean Dice Similarity Coefficient')
plt.ylim(0,1)
plt.show
#%%
box_precision_1 = sns.boxplot(x='Optimiser', y='val_Precision_1', hue='Augmentation',data=df_validation, showfliers=True, palette = my_palette)
plt.title("Bone segmentation precision (Validation)")
plt.ylabel('Precision')
plt.show
#%%
box_precision_2 = sns.boxplot(x='Optimiser', y='val_Precision_2', hue='Augmentation',data=df_validation, showfliers=True, palette = my_palette)
plt.title("Liver segmentation precision (Validation)")
plt.ylabel('Precision')
plt.legend(loc='lower left', title='Augmentation')
plt.show
#%%
box_precision_1 = sns.boxplot(x='Optimiser', y='val_Recall_1', hue='Augmentation',data=df_validation, showfliers=True, palette = my_palette)
plt.title("Bone segmentation recall (Validation)")
plt.ylabel('Recall')
plt.show
#%%
box_precision_2 = sns.boxplot(x='Optimiser', y='val_Recall_2', hue='Augmentation',data=df_validation, showfliers=True, palette = my_palette)
plt.title("Liver segmentation recall (Validation)")
plt.ylabel('Recall')
plt.show
#%%
box_precision_1 = sns.boxplot(x='Optimiser', y='val_F1_1', hue='Augmentation',data=df_validation, showfliers=True, palette = my_palette)
plt.title("Bone segmentation F1 (Validation)")
plt.ylabel('F1 Score')
plt.show
#%%
box_precision_2 = sns.boxplot(x='Optimiser', y='val_F1_2', hue='Augmentation',data=df_validation, showfliers=True, palette = my_palette)
plt.title("Liver segmentation F1 (Validation)")
plt.ylabel('F1 Score')
plt.show
#%%
sns.lineplot(x='Epoch No.', y='val_loss', hue='Recall Betas', size='Augmentation',
             style='Batchnorm', data=df_rf_val)#, ci=None)

# Decorations
plt.title('Recall-Favoured Validation Sensitivity w/Augmentation', fontsize=22)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.ylim(0, 1.0)
plt.show()

#%%
sns.lineplot(x='Epoch No.', y='loss', hue='Recall Weights', size='lr',
             style='Recall Betas', data=df_rf_test, ci=None)

# Decorations
plt.title('Testing Recall-Favoured Sensitivity w/Learning Rate', fontsize=22)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

#%%
sns.lineplot(x='Epoch No.', y='loss', hue='Recall Weights', size='Augmentation',
             style='Recall Betas', data=df_rf_test, ci=None)

# Decorations
plt.title('Testing Recall-Favoured Sensitivity w/Augmentation', fontsize=22)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

#%%
sns.lineplot(x='Epoch No.', y='val_loss', hue='Combo Loss Weight', size='Augmentation',
             style='Batchnorm', data=df_cl_val)#, ci=None)

# Decorations
plt.title('Combo-Loss Validation Sensitivity w/Augmentation & Batchnorm', fontsize=22)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#plt.ylim(0, 1.25)
plt.show()
#%% 
box = sns.boxplot(x='lr', y='val_F1_2', hue='Augmentation',
                   data=df_validation, showfliers=False)
plt.show()
#%%
box = sns.boxplot(x='Augmentation', y='val_loss', hue='Batchnorm',
                  data=df_cl_val, showfliers=False)
# Decorations
plt.title('Validation Recall-Favoured Sensitivity w/Learning Rate', fontsize=22)
plt.show()
#%%
sns.lineplot(x='Epoch No.', y='loss', hue='Combo Loss Weight', size='Augmentation',
             style='Batchnorm', data=df_cl_test, ci=None)

# Decorations
plt.title('Testing Combo-Loss Sensitivity w/Augmentation & Batchnorm', fontsize=22)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

#%%
grid = sns.FacetGrid(data=df_validation, col = 'Optimiser', row = "Loss Fn", hue='Dropout')
grid.map(plt.plot, 'Epoch No.','val_loss')
# Decorations
plt.title('Facetgrid', fontsize=22)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
#%%
# Plot
plt.figure(figsize=(12,10), dpi= 300)
corr = df_validation.corr(method='spearman')

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr.abs(), mask=mask, xticklabels=df_validation.corr().columns, yticklabels=df_validation.corr().columns,
            cmap='RdYlGn', center=0, annot=True, annot_kws={"fontsize":8})

# Decorations
plt.title('Correlogram', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()