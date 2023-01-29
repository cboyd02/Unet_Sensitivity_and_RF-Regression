"""
Miscellaneous functions used during training.

Some functions adapted from A/Prof Wolfgang Mayer
"""

import tensorflow.keras.backend as K
import gc, os
import tensorflow as tf
from tensorflow.config.experimental import get_memory_info
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import normalize


MASK_COLORS = [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
]

#%% Prevent GPU memory leakage during training by clearing session

def try_free_memory():
    K.clear_session()
    gc.collect()
    print("GPU Memory Usage (MB):{} of 11264 available".format(get_memory_info("GPU:0")['current']/1048576))
#%% Verify GPU will be used for training

def gpu_check():
    gpu_available = tf.config.list_physical_devices('GPU')
    is_cuda_gpu_available = tf.test.is_built_with_cuda()
    print(gpu_available, is_cuda_gpu_available)

#%% Count distribution of pixels in array - Useful for inverse class weighting of recall favoured loss function

def pixel_count(base_dir,  filetype, organ_string = ['Lumen', 'Atherosclerosis', 'Calcification'], dim = 512):
    ones_array = []
    zeros_array = []
    slice_total = 0
    normalised_distribution = []
    for organ in organ_string:    
        positive_array = []
        negative_array = []
        for root, folders, files in os.walk(base_dir):
            for file in files:
                if file.endswith(filetype):        
                    p = os.path.join(root, file)
                    if 'MASKS_DICOM' in p:
                        if organ in p:
                            ds = pydicom.dcmread(p)
                            pixeldata = ds.pixel_array
                            file_zeros = np.count_nonzero(pixeldata == 0)
                            file_ones = np.count_nonzero(pixeldata == 1)
                            if file_ones != 0:
                                positive_array.append(file_ones)
                            negative_array.append(file_zeros)
                            slice_total += 1
        organ_ones = np.sum(np.asarray(positive_array))
        organ_zeros = np.sum(np.asarray(negative_array))
        ones_array.append(organ_ones)
        zeros_array.append(organ_zeros)
    
    total = int(slice_total*dim*dim/len(organ_string))
    print(ones_array, zeros_array, total)    
    
    for i in range(len(organ_string)):
        positive = ones_array[i]
        if i ==0: bg_count = total - positive
        else: bg_count = bg_count - positive
        organ_fraction = total/positive
        print('The fraction of {} in masks is {}'.format(organ_string[i], organ_fraction))
        normalised_distribution.append(organ_fraction)

    bg_weight = total/bg_count
    normalised_distribution.insert(0,bg_weight)
    denom = sum(np.array(normalised_distribution))
    print(normalised_distribution/denom)
    
#%% Plot segmentation history
def plot_segm_history(history, metrics=["iou", "val_iou"], losses=["loss", "val_loss"]):
    """
    Plots variation in metrics across epochs, showing variation during training
    
    Args:
        history : Training history data
        metrics (list, optional): Which metrics to plot. Defaults to ["iou", "val_iou"].
        losses (list, optional): Which loss values to plot, training or validation. Defaults to ["loss", "val_loss"].
    """
    # summarize history for iou
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("metrics over epochs", fontsize=20)
    plt.ylabel("metric", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.legend(metrics, loc="center right", fontsize=15)
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(12, 6))
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle("loss over epochs", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    # plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    # plt.xticks(fontsize=35)
    plt.legend(losses, loc="center right", fontsize=15)
    plt.show()

#%% Convert binary mask to red
def mask_to_red(mask):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): original mask arrays
    
    Returns:
        numpy.ndarray: recoloured mask array
    """
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size, img_size)
    c2 = np.zeros((img_size, img_size))
    c3 = np.zeros((img_size, img_size))
    c4 = mask.reshape(img_size, img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)

#%% Convert mask to RGBA
def mask_to_rgba(mask, color="red"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): original mask arrays
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: recoloured mask array
    """    
    assert(color in MASK_COLORS)
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)

#%% Plot Images

def plot_imgs(
        org_imgs,
        mask_imgs,
        pred_imgs=None,
        nm_img_to_plot=10,
        figsize=4,
        alpha=0.5,
        color="red"):
    """
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.

    Args:
        org_imgs (numpy.ndarray): Array of arrays representing a collection of original images.
        mask_imgs (numpy.ndarray): Array of arrays representing a collection of mask images (grayscale).
        pred_imgs (numpy.ndarray, optional): Array of arrays representing a collection of prediction masks images.. Defaults to None.
        nm_img_to_plot (int, optional): How many images to display. Takes first N images. Defaults to 10.
        figsize (int, optional): Matplotlib figsize. Defaults to 4.
        alpha (float, optional): Transparency for mask overlay on original image. Defaults to 0.5.
        color (str, optional): Color for mask overlay. Defaults to "red".
    """
    assert(color in MASK_COLORS)

    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(
        nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=False
    )
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15)
        axes[0, 3].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(
                mask_to_rgba(
                    zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(pred_imgs),
                alpha=alpha,
            )
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(
                mask_to_rgba(
                    zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(mask_imgs),
                alpha=alpha,
            )
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.show()

#%% Zero pad mask
def zero_pad_mask(mask, desired_size):
    """
    Surround original mask file with 0's to change image shape
    
    Args:
        mask (numpy.ndarray): original mask arrays
        desired_size (int): linear array dimension required
    
    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask
#%% Reshape array
def reshape_arr(arr):
    """
    Reshape input array
    
    Args:
        arr (numpy.ndarray): original mask arrays
    
    Returns:
        numpy.ndarray: reshaped mask array
    """
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])

#%% Get colourmap
def get_cmap(arr):
    """
    
    Args:
        arr (numpy.ndarray): original mask arrays
    
    Returns:
        string: name description of colourmap being used for current array, based on dimensionality.
    """
    if arr.ndim == 3:
        return "gray"
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return "jet"
        elif arr.shape[3] == 1:
            return "gray"

#%% Get patches
def get_patches(img_arr, size=256, stride=256):
    """
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.
    
    Args:
        img_arr (numpy.ndarray): original mask arrays
        size (int, optional): dimension of patches to be extracted from img_arr. Defaults to 256.
        stride (int, optional): distance in pixels to move before extracting next patch. Defaults to 256.
    
    Returns:
        numpy.ndarray: arrays of "size" extracted from img_arr
    """    
    # check size and stride
    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(
                    img_arr[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size
                    ]
                )

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 3 or 4")

    return np.stack(patches_list)

#%% Plot patches
def plot_patches(img_arr, org_img_size, stride=None, size=None):
    """
    Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image

    Args:
        img_arr (numpy.ndarray):  original mask arrays
        org_img_size (tuple): dimension of original arrays
        stride (int, optional): dimension of patches to be extracted from img_arr. Defaults to None.
        size (int, optional): distance in pixels to move before extracting next patch. Defaults to None.

    """

    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1

#%% Reconstruct from patches

def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
    """
    Take array of patches and combine into org_img_size image, reconstructing images input to get_patches
    
    Args:
        img_arr (numpy.ndarray):  original mask arrays
        org_img_size (tuple): dimension of original arrays
        stride (int, optional): dimension of patches to be extracted from img_arr. Defaults to None.
        size (int, optional): distance in pixels to move before extracting next patch. Defaults to None.

    Returns:
        numpy.ndarray: [description]
    """
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = img_arr.shape[0] // (i_max ** 2)
    nm_images = img_arr.shape[0]

    averaging_value = size // stride
    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size,
                        layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1

        images_list.append(img_bg)

    return np.stack(images_list)
#%% Predict masks using model

def predict_masks(input_patients, model, size = 128, stride=128):
    """
    Parameters
    ----------
    input_patients : numpy.ndarray
        Array of patients to predict masks from
    model : .h5
        Trained model to use for prediction
    size : int, optional
        dimension of patches to predict. The default is 128.
    stride : int, optional
        distance to move before predicting next patch. The default is 128.

    Returns
    -------
    pred_full_masks_class : numpy.ndarray
        Array of predicted mask files.

    """
    patches = get_patches(input_patients, size=size, stride=stride)

    y_pred_patches = model.predict(patches)

    pred_full_masks_prob = reconstruct_from_patches(y_pred_patches, input_patients.shape[1:3])

    pred_full_masks_class = tf.keras.utils.to_categorical(np.argmax(pred_full_masks_prob, axis=-1),
                                                          num_classes=pred_full_masks_prob.shape[-1])

    return pred_full_masks_class

#%% Plot predicted masks against input images
def binary_plot_image_and_masks(title, patient_image, masks_true, masks_pred, labels):
    """
    Display patient image, ground truth mask and predicted mask in single figure for visual comparison.
    
    Parameters
    ----------
    title : string
        Header to put on each figure.
    patient_image : numpy.ndarray
        Arrray of patient image files.
    masks_true : numpy.ndarray
        Array of supplied binary contour masks.
    masks_pred : numpy.ndarray
        Array of predicted binary contour masks.
    labels : list
        List of strings to label each subfigure correctly.

    Returns
    -------
    None.

    """
    fig, axarr = plt.subplots(2, len(labels)+1, figsize=(10,10), sharex=False, sharey=False)
    axarr[0,0].imshow(normalize(patient_image[:,:,0]), cmap='gray')
    axarr[0,0].set_title(title)
    
    for channel, (ax, label) in enumerate(zip(axarr[0,1:], labels), 1):
      ax.imshow(masks_true[:,:,channel], cmap='gray')
      ax.set_title(str(label))
      
    for channel, (ax, label) in enumerate(zip(axarr[1,1:], labels), 1):
      ax.imshow(masks_pred[:,:,channel], cmap='gray')
      ax.set_title(str(label))

      
    axarr[1,0].imshow(masks_pred[:,:,0], cmap='gray')
    axarr[1,0].set_title('Background')
    fig.tight_layout()
    fig.subplots_adjust(top=0.975)
    plt.show()
    
#%% Plot predicted masks against input images

def plot_image_and_masks(title, patient_image, masks_true, masks_pred, labels, cmap=ListedColormap(["black", "white", "red", "green"])):
    """
    Display patient image, ground truth mask and predicted mask in single figure for visual comparison.
    
    Parameters
    ----------
    title : string
        Header to put on each figure.
    patient_image : numpy.ndarray
        Arrray of patient image files.
    masks_true : numpy.ndarray
        Array of supplied binary contour masks.
    masks_pred : numpy.ndarray
        Array of predicted binary contour masks.
    labels : list
        List of strings to label each subfigure correctly.
    cmap : List
        List of strings identifying which colours to use for display. Defaults to ["black", "white", "red", "green"].

    Returns
    -------
    None.

    """
    def show_mask(mask_true, mask_pred, ax):
      tp = mask_true * mask_pred
      fp = (1 - mask_true) * mask_pred
      fn = mask_true * (1 - mask_pred)
      ax.imshow(tp+2*fn+3*fp, cmap=cmap)

    fig, axarr = plt.subplots(1, len(labels)+2, figsize=(10,10), sharex=False, sharey=False)
    axarr[0].imshow(patient_image[:,:,0], vmin = -1024, vmax = 1023, interpolation = 'none',  cmap='gray')
    axarr[0].set_title(title)
    
    for channel, (ax, label) in enumerate(zip(axarr[1:], labels), 1):
      show_mask(masks_true[:,:,channel], masks_pred[:,:,channel], ax)
      ax.set_title(str(label))
      #ax.legend(loc='best', title=str(label))
      
    show_mask(masks_true[:,:,0], masks_pred[:,:,0], axarr[-1])
    axarr[-1].set_title('Background')
    fig.tight_layout()
    fig.subplots_adjust(top=0.975)
    plt.show()