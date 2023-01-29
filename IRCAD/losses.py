"""
Custom loss functions used during model training, including references for design and use.
https://doi.org/10.1016/j.media.2021.102035 - Review of loss functions in class imbalanced medical imaging segmentation
"""
#%%
from tensorflow.keras import backend as K
import tensorflow as tf

def loss_recall_favoured(num_classes, ClassWeights, beta = 2, BG_weight=0.5):
    """
    Parameters
    ----------
    num_classes : int
        Number of classes, including background, requiring segmentation.
    ClassWeights : List
        List of floats to weight each class loss by, used to correct for class imbalance.
    beta : int, optional
        Recall favoured loss argument. The default is 2.
    BG_weight : float, optional
        Weight for background. The default is 0.5.

    Returns
    -------
    The recall favoured loss between the two tensors for ecah class.

    """
    if len(ClassWeights) != num_classes:
        raise IndexError("Class weights do not equal number of classes")
    Weights = ClassWeights
    Betas = [beta for x in range(num_classes)]  #values of 1.0 will give soft Dice loss
    Betas[0]= BG_weight #0.5 to weight precision for background
    def loss(y_true, y_pred):
        """
        Recall favoured loss function for semantic segmentation.

        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor

        # Returns
            The Recall favoured loss between the two tensors for the given class.

        # References
            - https://arxiv.org/pdf/1803.11078.pdf

        """
        tp = K.sum(y_true * y_pred, axis=[0,1,2]) #?flatten, reshape or cast?
        fp = K.sum(y_pred, axis=[0,1,2]) - tp
        fn = K.sum(y_true, axis=[0,1,2]) - tp
        W1 = [1.0+x**2 for x in Betas]
        W2 = [x**2 for x in Betas]
        return K.mean(Weights*(1.0 - (W1*tp + K.epsilon()) / ( W1*tp +  W2*fn + fp + K.epsilon())))
    return loss

def dice_coef_loss(ignore_bg=True):
    def loss(y_true, y_pred, smooth=1):
        """
        Dice similarity coefficient loss function for multiclass semantic segmentation.

        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 1

        # Returns
            The Dice loss between the two tensors for the given class.

        # References
            - https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py
            - https://github.com/robinvvinod/unet/blob/master/losses.py
            - https://arxiv.org/pdf/1803.11078.pdf
        
        """
        if ignore_bg==False: 
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
        else:
            y_true_f = K.flatten(y_true[...,1:])
            y_pred_f = K.flatten(y_pred[...,1:])
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return loss

def combo_loss(ignore_bg=True, cce_weight = 0.2):
    def loss(y_true, y_pred, smooth=1):
        """
        Weighted combination of categorical cross entropy loss and dice similarity coefficient loss for semantic segmentation.

        # Arguments
            ignore_bg: Whether to include background class in calculation of comboloss. Default is True.
            cce_weight: Weight (w) of categorical cross entropy, dice coefficient weighted (1-w).
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 1

        # Returns
            The Dice loss between the two tensors for the given class.

        # References
            - https://gchlebus.github.io/2018/02/18/semantic-segmentation-loss-functions.html
            - https://arxiv.org/pdf/1707.03237.pdf
            - https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py
            - https://github.com/robinvvinod/unet/blob/master/losses.py
        """
        
        log_p = -tf.math.log(tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon()))
        reduced_loss =  tf.math.reduce_sum(y_true*log_p, axis = -1)
        cce_loss = tf.math.reduce_mean(reduced_loss) * cce_weight
       
        if ignore_bg==False: 
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
        else:
            y_true_f = K.flatten(y_true[...,1:])
            y_pred_f = K.flatten(y_pred[...,1:])
        intersection = K.sum(y_true_f * y_pred_f)
        dsc_loss =  (1 - K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))*(1-cce_weight)
        return dsc_loss + cce_loss
    return loss

def tversky_loss(smooth=1, alpha=0.7):
    def loss(y_true, y_pred):    
        """
        Tverskty loss function for semantic segmentation.
    
        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 1
            alpha: Loss function parameter. Default is 0.7

        # Returns
            The Tversky loss between the two tensors for the given class.
    
        # References
            - https://arxiv.org/pdf/1810.07842.pdf - Suggests gamma = 1.33
            - https://github.com/robinvvinod/unet/blob/master/losses.py
        """
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.7
        return 1 - (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
    return loss

def focal_tversky_loss(smooth=1, alpha=0.7, gamma=0.75):
    def loss(y_true, y_pred):    
        """
        Focal tverskty loss function for semantic segmentation. Modified version of tversky loss, raised to power gamma.
    
        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 1
            alpha: Loss function parameter. Default is 0.7
            gamma: Loss function parameter. Default is 0.75 (1.333 suggested in references, but formula here doesn't use  1/gamma)
        # Returns
            The Focal tversky loss between the two tensors for the given class.
    
        # References
            - https://arxiv.org/pdf/1810.07842.pdf - Suggests gamma = 1.33
            - https://github.com/robinvvinod/unet/blob/master/losses.py
            - https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
        """
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.7
        return K.pow((1-(true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)), gamma)
    return loss

def jaccard_distance():
    def loss(y_true, y_pred, smooth=100):
        """Jaccard distance for semantic segmentation.
    
        The loss has been modified to have a smooth gradient as it converges on zero.
        This has been shifted so it converges on 0 and is smoothed to avoid exploding
        or disappearing gradient.
    
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 100.
    
        # Returns
            The Jaccard distance between the two tensors.
    
        # References
            - [What is a good evaluation measure for semantic segmentation?](
               http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    
        """
        y_true_f = K.flatten(y_true[...,1:])
        y_pred_f = K.flatten(y_pred[...,1:])       
        intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
        sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac)# * smooth
    return loss

def cce_loss(y_true, y_pred):
    """
    Categorical cross entropy loss function for semantic segmentation.

    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
                                                         
    # Returns
        The Categorical cross entryop loss between the two tensors for the given class.

    # References
        - https://gchlebus.github.io/2018/02/18/semantic-segmentation-loss-functions.html
    """
    log_p = -tf.math.log(tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon()))
    loss =  tf.math.reduce_sum(y_true*log_p, axis = -1)
    return tf.math.reduce_mean(loss)