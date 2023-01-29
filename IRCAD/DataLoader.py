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
        """
        Walk through IRCAD dataset and collect patient image files and masks in structured dataset
        
        Parameters
        ----------
        root_dir : pathstring
            Root directory of IRCAD data.
        organs : list
            a list of names of the contour mask folders corresponding to the organs the model is being trained to segment.

        Returns
        -------
        None.

        """
        file_extension="" #ignore files with extensions, as IRCAD DICOMs don't have any
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
                        rs = re.match(r".*MASKS_DICOM\\(.*)\\.*", str(p))
                        patient_images[file][rs.groups()[0]] = p

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
        """
        Normalize input data to be between 0 and 1
        """
        arr = img.copy().astype(np.float)
        M = np.float(np.max(img))
        if M != 0:
            arr *= 1./M
        return arr

    def transform(self, image, mask, augment_type='modify', crop_dim=128):
         """
         Perform all data augmentation. Additional augmentations can be added from Albumentations.
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

    def augmented_dataset(self, training_imgs, training_masks, aug_samples, augment_type):
        """
        Generate an augmented dataset based on input images and add to the original un-augmented dataset
        """
        aug_pt_imgs = []
        aug_pt_masks = []
    
        for k in range(aug_samples):
             aug_imgs, aug_masks = self.transform(training_imgs, training_masks, augment_type)
             aug_pt_imgs.append(aug_imgs)
             aug_pt_masks.append(aug_masks)
    
        aug_pt_imgs.append(training_imgs) #Add back un-augmented images to final
        aug_pt_masks.append(training_masks)
    
        training_imgs = np.asarray(aug_pt_imgs, dtype='float32')
        training_masks = np.asarray(aug_pt_masks, dtype='float32')
    
        return training_imgs, training_masks


    def get_data(self, noisy=False, new_dim=512, verbose=False, figures=False, augment=False):
        """
        Step through a single directory of IRCAD DICOM patient and mask files, collecting each.
        Return data as formatted numpy arrays ready for use in model development
        """
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

                for k in range(len(full_img_x)):
                    for j in range(8):
                        img_x, img_y = self.transform(full_img_x[k], full_img_y[k], 'crop', new_dim)

                        if verbose: print(np.count_nonzero(img_x, axis=(0,1)),np.count_nonzero(img_y, axis=(0,1)))

                        im_X.append(img_x)
                        im_Y.append(img_y)

            else:
                for j in range(8):
                    img_x, img_y = self.transform(full_img_x, full_img_y, 'crop', new_dim)

                    if verbose: print(np.count_nonzero(img_x, axis=(0,1)),np.count_nonzero(img_y, axis=(0,1)))

                    im_X.append(img_x)
                    im_Y.append(img_y)

        # Displaying some samples of the input
        if figures:
            
            lines = min(10,len(im_X))

            fig, axarr = plt.subplots(lines, 2, figsize=(60,lines*20), sharex=True, sharey=False)

            for i in range(0,lines):

                axarr[i,0].imshow(im_X[i], cmap='gray')
                axarr[i,1].imshow(im_Y[i], cmap='rainbow')

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
        
        input_onePt = np.expand_dims(shuffleX,-1)
        target_onePt = to_categorical(shuffleY,num_classes=self.ROI_classes+1) #+1 for bg
        
        return input_onePt, target_onePt

    def patient_dirs(root_dir, patient_folder_string):
        """
        os.walk through root_dir and collect all paths containing patient_folder_string (the DICOM files from IRCAD). 
        Used for identifying patients in IRCAD root directory
        """
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
        """
        Loop through all IRCAD patient paths, running get_data on each and combining into two image and masks datasets.
        
        Parameters
        ----------
        sample_size : int
            Total number of patients to import from those identified by patient_dirs
        organs : list
            List of organ names to be imported for training.
        patient_paths : list
            List of paths to patient datasets.
        verbose : boolean, optional
            Boolean to run print statements during execution. The default is False.
        figures : boolean, optional
            Boolean to generate and display figures during execution. The default is False.
        augment : boolean, optional
            Whether perform augmentations during training. The default is False.
        patchsize : int, optional
            Size to crop image and mask files to during training. The default is 128.

        Returns
        -------
        input_allPt : numpy array
            Multidimensional numpy array where each patient image dataset is stored.
        target_allPt : numpy array
            Multidimensional numpy array where each set of patient masks are stored.

        """
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
            input_all_list.append(input_onePt)
            target_all_list.append(target_onePt)

            print(len(input_onePt), len(target_onePt), type(input_onePt))

        input_allPt = np.asarray(input_all_list)
        target_allPt = np.asarray(target_all_list)

        print(type(input_allPt), input_allPt[0].shape, type(target_allPt), target_allPt[0].shape)
        return input_allPt, target_allPt