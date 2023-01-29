import os, sys, re
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
checkpoint_dir = os.path.join(base_dir, 'checkpoints')
print(checkpoint_dir)

import albumentations as A
import pydicom as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
#%%
root_dir = r"[PLACEHOLDER PATH] - Map to PatientData directory containing IRCAD dataset"
organs = ['bone', 'liver'] # ['skin', 'bone', 'liver'] #skin has to go first to avoid overlay issues
patient_folder_string = "3Dircadb"
#%%
def patient_dirs(root_dir, patient_folder_string):
    """os.walk through root_dir and collect paths containing patient_folder_string (the DICOM files from IRCAD)"""
    patient_paths = []
    patient_folders = os.listdir(root_dir)
    patients = [string for string in patient_folders if patient_folder_string in string]

    for root, dirs, files in os.walk(root_dir):
        for patient_folders in dirs:
            q = os.path.join(root, patient_folders)
            for i in range(len(patients)):
                if q.endswith(patients[i]):
                    patient_paths += [q]
    print(len(patient_paths), patient_paths)
    return patient_paths                      

def transform(image, mask, augment_type='modify', crop_dim=128):
     """Perform all data augmentation. Additional augmentations can be added from Albumentations.
     If augment_type is 'crop', crop input images to crop_dim instead of augmenting.
     """
     if augment_type=='modify':
         augment = A.Compose([
             # Pixel level transforms automatically applied only to images,
             # Spatial transforms applied to both: https://albumentations.ai/docs/getting_started/transforms_and_targets/
             
             ### Pixel Transforms ###
             A.Downscale(scale_min=0.9, scale_max=0.9, p=0.25),
             A.MotionBlur(blur_limit=5, p=0.25),
             A.GaussNoise(var_limit=(10, 50), mean=0, p=0.25),
             
             ###Spatial Transforms ###
             A.Rotate(limit=20, value=-2048, mask_value=0, p=0.25), #Apply small rotation and then pad with air
             ])
     elif augment_type=='crop':
         augment = A.Compose([
            A.CropNonEmptyMaskIfExists(crop_dim, crop_dim)
            ])
     else:
         print("Invalid transform type provided - Please input valid transform type")

     transformed = augment(image=image, mask=mask)
     transformed_image = transformed['image']
     transformed_mask = transformed['mask']
     return transformed_image, transformed_mask

def augmented_dataset(training_imgs, training_masks, aug_samples, augment_type):
    """"Generate an augmented dataset based on input images and add to the original un-augmented dataset"""
    aug_pt_imgs = []
    aug_pt_masks = []

    for k in range(aug_samples):
         aug_imgs, aug_masks = transform(training_imgs, training_masks, augment_type)
         aug_pt_imgs.append(aug_imgs)
         aug_pt_masks.append(aug_masks)

    aug_pt_imgs.append(training_imgs) #Add back un-augmented images to final
    aug_pt_masks.append(training_masks)

    training_imgs = np.asarray(aug_pt_imgs, dtype='float32')
    training_masks = np.asarray(aug_pt_masks, dtype='float32')

    return training_imgs, training_masks

def img_visualise(array_img):
    """Display input DICOM images with grey colourmap"""
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(array_img, cmap='Greys')

def mask_visualise(array_img, organs):
    """Display input masks separately, as binary images"""
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(array_img, vmin=0, vmax=len(organs))
    
def DataGenerator(root_dir, organs):
    """
    Walk through IRCAD dataset and collect patient image files and masks in structured dataset
    """
    file_extension=""
    patient_images = {}


    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(file_extension):
                if 'PATIENT_DICOM' in root:
                    if not patient_images.get(file,None):
                        patient_images[file] = {}
                    p = os.path.join(root,file)
                    patient_images[file]['real'] = p
                    #print(patient_images[file])

                elif 'MASKS_DICOM' in root:
                    if not patient_images.get(file,None):
                        patient_images[file] = {}
                    p = os.path.join(root,file)
                    
                    rs = re.match(r".*MASKS_DICOM\\(.*)\\.*", str(p))
                    patient_images[file][rs.groups()[0]] = p
              
        X = []
        Y = []
        organ_images = {}
    
        ##Import patient DICOM paths into array##
        for k,v in patient_images.items():
            for k1, v1 in v.items():
                if k1 == 'real':
                    X.append(v['real'])

        ##Importing mask DICOM paths into array##
        for k,v in patient_images.items():
            for k1,v1 in v.items():
                if k1 not in organ_images:
                    organ_images[k1] = []
                organ_images[k1].append(v1)
        Y = dict((k,organ_images[k]) for k in organs if k in organ_images)
    
        im_X = []
        im_Y = []
        segment = []
                        

    for k,v in Y.items():
        organ_no = list(Y.keys()).index(str(k))
        for t in range(len(v)):

            mask = pd.read_file(v[t]).pixel_array.astype(np.uint8)
            mask[mask > 0] = organ_no + 1
            
            if organ_no == 0:
                segment.append(mask)
            else:
                segment[t] = np.fmax(segment[t], mask)

    for i in tqdm(range(len(X))):
         full_img_x = pd.read_file(X[i]).pixel_array
         full_img_y = segment[i]

         im_X.append(full_img_x)
         im_Y.append(full_img_y)
    print(len(im_Y))
    
    return np.asarray(im_X),  np.asarray(im_Y)

#%%
"""
Import IRCAD files and run DataGenerator, including augmentation
then save as .npz so datasets dont need to be augmented 'online' during training
"""

paths = patient_dirs(root_dir, patient_folder_string)
input_all_list, target_all_list = [], []

for pts in range(len(paths)):
    X_pt, Y_pt = DataGenerator(paths[pts], organs)
    aug_copies = 2
    input_onePt, target_onePt = [] , [] #empty variables for each patient
    img_X_pt, img_Y_pt = [], [] # Each augmented and cropped patch of image and corresponding mask/s
    
    for i in range(len(X_pt)):
        X_aug, Y_aug = augmented_dataset(X_pt[i], Y_pt[i], aug_copies, 'modify')
        
        for j in range(2):#len(X_aug)):
            for k in range(8):
                img_x, img_y = transform(X_aug[j], Y_aug[j], 'crop', 128)    
                img_X_pt.append(img_x)
                img_Y_pt.append(img_y)           
        
    input_onePt = np.expand_dims(img_X_pt,-1)
    target_onePt = to_categorical(img_Y_pt,num_classes=len(organs)+1)
    
    input_all_list.append(input_onePt)
    target_all_list.append(target_onePt)

#%% Convert from list to numpy array for saving as .npz
input_allPt = np.asarray(input_all_list, dtype=object)
target_allPt = np.asarray(target_all_list, dtype=object)
#%% Generate descriptive string for naming and save to checkpoint directory
aug_set = os.path.join(checkpoint_dir, str(len(organs))+'organ-'+str(aug_copies)+'copies-'+len(paths)+'patients-aug_data')
np.savez(aug_set, input_allPt, target_allPt)





