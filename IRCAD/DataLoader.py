# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:03:10 2021

@author: chris
"""
import numpy as np
from random import shuffle
import pydicom
import os, re
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm

class PatientData(object):
    """

    PatientData looks for files of 3DIRCAD database.
    This database contains DICOM files and data is split into folders.
    PATIENT_DICOM folder contains original original CT Images
    MASKS_DICOM contains a list of several folders. Each folder is named
    according to the organ highlighted in the masks of the files within.
    During PatientData initialization, it will look for the folder pointed at
    root_dir and will load files named with same name on MASKS_DICOM/<organ name>/*

    """

    def __init__(self, root_dir, organs):

        file_extension=""
        patient_images = {}
        ROI_classes = len(organs)

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
                        #print(p)
                        rs = re.match(r".*MASKS_DICOM\\(.*)\\.*", str(p))
                        patient_images[file][rs.groups()[0]] = p
                        #print(p)
        self.X = []
        self.Y = []
        self.organ_images = {}

        ##Import patient DICOM paths into array##
        for k,v in patient_images.items():
            for k1, v1 in v.items():
                if k1 == 'real':
                    self.X.append(v['real'])

        ##Importing mask DICOM paths into array##
        for k,v in patient_images.items():
            #print(k)
            for k1,v1 in v.items():
                if k1 not in self.organ_images:
                    self.organ_images[k1] = []
                self.organ_images[k1].append(v1)
        self.Y = dict((k,self.organ_images[k]) for k in organs if k in self.organ_images)

        self.root_dir = root_dir
        self.patient_images = patient_images
        self.ROI_classes = ROI_classes

    def normalize(self, img):
        arr = img.copy().astype(np.float)
        M = np.float(np.max(img))
        if M != 0:
            arr *= 1./M
        return arr


    def add_gauss_noise(self, inp, expected_noise_ratio=0.05):
        image = inp.copy()
        if len(image.shape) == 2:
            row,col= image.shape
            ch = 1
        else:
            row,col,ch= image.shape
        mean = 0.
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col)) * expected_noise_ratio
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy

    def transform(self, image, mask, augment_type='modify', crop_dim=128):
         #print(type(image), type(mask))
         if augment_type=='modify':
             augment = A.Compose([
                 A.OneOf([
                 A.Flip(p=1),
                 A.GaussNoise(var_limit=(0.0,1.0),mean=0, p=1),
                 A.RandomRotate90(p=1.0),
                 # A.Blur(blur_limit=(1,5),p=1),
                 # A.ElasticTransform(p=1.0),],
                 ], p=1)
                 #A.GaussianBlur(blur_limit=(3,7), sigma_limit=0 ,p=0.2),
                 #A.Downscale(p=0.2)
                 #A.RandomCrop(width=256, height=256),
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
    #https://albumentations.ai/docs/getting_started/mask_augmentation/

    def augmented_dataset(self, training_imgs, training_masks, aug_samples, augment_type):
        aug_pt_imgs = []
        aug_pt_masks = []
        # training_imgs=training_imgs.tolist()
        # training_masks=training_masks.tolist()
        # print(len(training_imgs),np.asarray(training_imgs, dtype=object).shape)
        for k in range(aug_samples):
             aug_imgs, aug_masks = self.transform(training_imgs, training_masks, augment_type)
             # for i in range(len(training_imgs)):
             #    aug_pt_imgs = []
             #    aug_pt_masks = []

             #    for j in range(len(training_imgs[i])):
             #        if j%50==0:
             #            print(i,j)
             #        aug_imgs, aug_masks = self.transform(training_imgs[i][j], training_masks[i][j], augment_type)
                     # aug_pt_imgs.append(aug_imgs)
                     # aug_pt_masks.append(aug_masks)
             aug_pt_imgs.append(aug_imgs)
             aug_pt_masks.append(aug_masks)

        aug_pt_imgs.append(training_imgs) #Add back un-augmented images to final
        aug_pt_masks.append(training_masks)

        # print("augment {} completed".format(k))
        training_imgs = np.asarray(aug_pt_imgs, dtype='float32')#.astype('float32')#, dtype=object)
        training_masks = np.asarray(aug_pt_masks, dtype='float32')#.astype('float32')#, dtype=object)


        return training_imgs, training_masks


    def get_data(self, noisy=False, new_dim=512, verbose=False, figures=False, augment=False): #get_data(self, noisy=False, split_part=0.5, new_dim=None, verbose=False, figures=False):
        im_X = []
        im_Y = []
        segment = []


        for k,v in self.Y.items():
            organ_no = list(self.Y.keys()).index(str(k))
            for t in range(len(v)):
                if verbose: print("organ: {k} slice: {t}".format(k=k, t=t))
                mask = pydicom.read_file(v[t]).pixel_array.astype(np.uint8)
                mask[mask > 0] = organ_no + 1
                if organ_no == 0:
                    segment.append(mask)
                else:
                    segment[t] = np.fmax(segment[t], mask)

        for i in tqdm(range(len(self.X))):
            full_img_x = pydicom.read_file(self.X[i]).pixel_array
            full_img_y = segment[i]
            full_img_x = self.normalize(full_img_x) #replace this normalize with HU normalise later

            if verbose: print("max of segment {i} is: {seg}".format(i=i, seg= segment[i].max()))

            if augment:
                full_img_x, full_img_y = self.augmented_dataset(full_img_x, full_img_y, 1, 'modify')
                #print(len(full_img_x), type(full_img_x[1]), full_img_x[1].shape)
                for k in range(len(full_img_x)):
                    for j in range(8):
                        img_x, img_y = self.transform(full_img_x[k], full_img_y[k], 'crop', new_dim)

                        #Reject images that are empty (or near empty) of organs
                        #if np.count_nonzero(img_y, axis=(0,1)) < 1: # 82   #0.5% of 128*128 array
                            #continue
                        if verbose: print(np.count_nonzero(img_x, axis=(0,1)),np.count_nonzero(img_y, axis=(0,1)))
                        #positive_pix_count += np.count_nonzero(img_y, axis=(0,1))

                        im_X.append(img_x)
                        im_Y.append(img_y)

                        # if j == 0 & k == 0:
                        #     print("X's:",type(img_x), img_x.shape, type(im_X), len(im_X), type(im_X[0]), im_X[0].shape)
                        #     print("Y's:",type(img_y), img_y.shape, type(im_Y), len(im_Y), type(im_Y[0]), im_Y[0].shape)

            #Iteratively import patches of each image based on output dimensions
            # for j,k in zip(range(int(int(512/new_dim))),range(int(int(512/new_dim)))):
                # x_start = j*new_dim
                # y_start = k*new_dim

                # img_x = full_img_x[x_start:x_start+new_dim, y_start: y_start+new_dim]
                # img_y = full_img_y[x_start:x_start+new_dim, y_start: y_start+new_dim]
            else:
                for j in range(8):
                    img_x, img_y = self.transform(full_img_x, full_img_y, 'crop', new_dim)

                    #Reject images that are empty (or near empty) of organs
                    #if np.count_nonzero(img_y, axis=(0,1)) < 1: # 82   #0.5% of 128*128 array
                        #continue
                    if verbose: print(np.count_nonzero(img_x, axis=(0,1)),np.count_nonzero(img_y, axis=(0,1)))
                    #positive_pix_count += np.count_nonzero(img_y, axis=(0,1))

                    im_X.append(img_x)
                    im_Y.append(img_y)

                    # print("X's:",type(img_x), img_x.shape, len(im_X), type(im_X[0]), im_X[0].shape)
                    # print("Y's:",type(img_y), img_y.shape, len(im_Y), type(im_Y[0]), im_Y[0].shape)


        # Displaying some samples of the input
        if figures:
            # print("X's:",type(im_X), len(im_X), type(im_X[0]), im_X[0].shape)
            lines = min(10,len(im_X))

            fig, axarr = plt.subplots(lines, 2, figsize=(60,lines*20), sharex=True, sharey=False)

            for i in range(0,lines):

                axarr[i,0].imshow(im_X[i], cmap='gray') #.reshape(new_dim, new_dim)
                axarr[i,1].imshow(im_Y[i], cmap='rainbow') #.reshape(new_dim, new_dim)

                for x in range(2):
                    axarr[i,x].axis('off')
            fig.tight_layout()
            fig.subplots_adjust(top=0.975)
            plt.show()

                ##Check patient images and masks are paired##
        if len(im_X) != len(im_Y):
            raise Exception("number of input images (%d) does not match number of training samples (%d)" %
                            (len(im_X),len(im_Y)))

        indexes = list(range(len(im_X)))
        shuffle(indexes)

        shuffleX = [im_X[c] for c in indexes]
        shuffleY = [im_Y[c] for c in indexes]
        #print(type(shuffleX),type(shuffleY))
        input_onePt = np.expand_dims(shuffleX,-1)
        target_onePt = to_categorical(shuffleY,num_classes=self.ROI_classes+1) #+1 for bg
        #print(type(input_onePt), type(target_onePt))
        return input_onePt, target_onePt

    def patient_dirs(root_dir, patient_folder_string):
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

    def pt_mask_import(sample_size, organs, patient_paths, verbose=False, figures=False, augment=False, patchsize=128):
        ###############################
        # Declare variables used in model
        ###############################
        pts = 0
        input_allPt = []
        target_allPt = []
        input_all_list = []
        target_all_list = []
        ###############################
        # Loop patient import over all patients
        ###############################

        for pts in range(sample_size):
            #############################################
            #Import data for single patient
            #############################################
            print("##### Importing Patient {} #####".format(pts+1))
            pat_data = PatientData(patient_paths[pts], organs)

            #############################################
            #Prepare data
            #############################################

            data = pat_data.get_data(False, verbose=verbose, figures=figures, augment=augment, new_dim=patchsize)
            input_onePt, target_onePt = map(lambda x : x, data)

            #############################################
            #Create single input and target array's containing all patients
            #############################################
            #print(type(input_onePt), type(input_all_list))
            input_all_list.append(input_onePt)
            target_all_list.append(target_onePt)

            print(len(input_onePt), len(target_onePt), type(input_onePt))


        input_allPt = np.asarray(input_all_list)
        target_allPt = np.asarray(target_all_list)

        print(type(input_allPt), input_allPt[0].shape, type(target_allPt), target_allPt[0].shape)#, target_allPt[2][0][64], target_allPt[2][64][0], target_allPt[2][64][0].shape)
        # plt.figure()
        # plt.imshow(target_allPt[2][0])
        # plt.show()
        return input_allPt, target_allPt