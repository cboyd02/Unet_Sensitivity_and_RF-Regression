from tensorflow.keras import backend as K
import numpy as np
#Adapted from M. McDonnell + W. Mayer
verbose = False

def metric_tf_initialise(num_classes):
    
    class_indexes = np.arange(num_classes)
    metrics = ['accuracy', 'categorical_accuracy',
              metric_multiclass_jaccard_coef_categorical_int(class_indexes),
              metric_multiclass_dice_coef_categorical_int(class_indexes)]
    for metric_fn in [metric_multiclass_jaccard_coef_categorical_int,
                      metric_multiclass_dice_coef_categorical_int, 
                      metric_multiclass_recall,
                      metric_multiclass_precision]:
        metrics.extend([metric_fn([i]) for i in class_indexes])

    return metrics


def metric_multiclass_jaccard_coef_categorical_int(DesiredClasses):
    #https://ieeexplore.ieee.org/document/9116807
    def metric(y_true, y_pred):
        metric = 0.0
        class_preds = K.argmax(y_pred,axis=-1)
        for i, TargetClass in enumerate(DesiredClasses):
            if TargetClass==0 and len(DesiredClasses)>1: continue #remove BG from mean calculation - Keep for class specific calculation
            predicted_mask = K.cast(K.equal(class_preds, TargetClass), 'float32')
            actual_mask = K.cast(K.equal(y_true[:,:,:,TargetClass], 1), 'float32')
            #get the intersection and union
            total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1,2])
            total_class_sum = K.sum(actual_mask + predicted_mask, axis=[0,1,2])
            #calculate metric for this class
            metric += (total_class_intersection + K.epsilon()) / (total_class_sum - total_class_intersection + K.epsilon())
            #print(i, TargetClass, DesiredClasses)
        if len(DesiredClasses)>1:
            #print(i, TargetClass, len(DesiredClasses), metric, metric/(len(DesiredClasses)-1))
            return metric/(len(DesiredClasses)-1)
        else:
            return metric
        #return metric/len(DesiredClasses)
    
    if verbose:
        print("The length of DesiredClasses for Jaccard is: ", len(DesiredClasses)) 
    if len(DesiredClasses)>1:
        metric.__name__ = 'JI_{}'.format('mean')
    else:
        metric.__name__ = 'JI_{}'.format(str(DesiredClasses[0]))
    return metric

def metric_multiclass_dice_coef_categorical_int(DesiredClasses):
    def metric(y_true, y_pred):
        metric = 0.0
        class_preds = K.argmax(y_pred,axis=-1)
        for i, TargetClass in enumerate(DesiredClasses):
            if TargetClass==0 and len(DesiredClasses)>1: continue #remove BG from mean calculation - Keep for class specific calculation
            predicted_mask = K.cast(K.equal(class_preds, TargetClass), 'float32')
            actual_mask = K.cast(K.equal(y_true[:,:,:,TargetClass], 1), 'float32')
            #get the intersection and union
            total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1,2])
            total_class_sum = K.sum(actual_mask + predicted_mask, axis=[0,1,2])
            #calculate metric for this class
            metric += ((2. * total_class_intersection) + K.epsilon()) / (total_class_sum + K.epsilon())
        if len(DesiredClasses)>1:
            return metric/(len(DesiredClasses)-1)
        else:
            return metric
        #return metric/len(DesiredClasses)
    if verbose:
        print("The length of DesiredClasses for Dice is: ", len(DesiredClasses))
    if len(DesiredClasses)>1:
        metric.__name__ = 'F1_{}'.format('mean')
    else:
        metric.__name__ = 'F1_{}'.format(str(DesiredClasses[0]))
    return metric

def metric_multiclass_recall(DesiredClasses):
    def metric(y_true, y_pred):
        metric = 0.0
        class_preds = K.argmax(y_pred,axis=-1)
        for i, TargetClass in enumerate(DesiredClasses):
            if TargetClass==0 and len(DesiredClasses)>1: continue #remove BG from mean calculation - Keep for class specific calculation
            predicted_mask = K.cast(K.equal(class_preds, TargetClass), 'float32') #elementwise matching and conversion from tensor to float32
            actual_mask = K.cast(K.equal( y_true[:,:,:,TargetClass], 1), 'float32')
            tp = K.sum(actual_mask * predicted_mask, axis=[0,1,2])
            fp = K.sum(predicted_mask, axis=[0,1,2]) - tp
            fn = K.sum(actual_mask, axis=[0,1,2]) - tp
            metric += (tp + K.epsilon()) / (tp+fn + K.epsilon()) #epsilon (10^-7) added to avoid div0 errors
        if len(DesiredClasses)>1:
            return metric/(len(DesiredClasses)-1)
        else:
            return metric
        #return metric/len(DesiredClasses)
    if verbose:
        print("The length of DesiredClasses for Recall is: ", len(DesiredClasses))
    if len(DesiredClasses)>1:
        metric.__name__ = 'Recall_{}'.format('mean')
    else:
        metric.__name__ = 'Recall_{}'.format(str(DesiredClasses[0]))
    return metric

def metric_multiclass_precision(DesiredClasses):
    def metric(y_true, y_pred):
        metric = 0.0
        class_preds = K.argmax(y_pred,axis=-1)
        for i, TargetClass in enumerate(DesiredClasses):
            if TargetClass==0 and len(DesiredClasses)>1: continue #remove BG from mean calculation - Keep for class specific calculation
            predicted_mask = K.cast(K.equal(class_preds, TargetClass), 'float32')
            actual_mask = K.cast(K.equal(y_true[:,:,:,TargetClass], 1), 'float32')
            tp = K.sum(actual_mask * predicted_mask, axis=[0,1,2])
            fp = K.sum(predicted_mask, axis=[0,1,2]) - tp
            fn = K.sum(actual_mask, axis=[0,1,2]) - tp
            metric += (tp + K.epsilon()) / (tp+fp + K.epsilon())
        if len(DesiredClasses)>1:
            return metric/(len(DesiredClasses)-1)
        else:
            return metric
        #return metric/len(DesiredClasses)
    if verbose:
        print("The length of DesiredClasses for Precision is: ", len(DesiredClasses))
    if len(DesiredClasses)>1:
        metric.__name__ = 'Precision_{}'.format('mean')
    else:
        metric.__name__ = 'Precision_{}'.format(str(DesiredClasses[0]))
    return metric

""" Unused metrics - single class versions of multiclass provided above"""
# def iou(y_true, y_pred, smooth=1.):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

    
# def jaccard_coef(y_true, y_pred):
#     intersection = K.sum(y_true * y_pred)
#     union = K.sum(y_true + y_pred)
#     jac = (intersection + 1.) / (union - intersection + 1.)
#     return K.mean(jac)


# def threshold_binarize(x, threshold=0.5):
#     ge = tf.greater_equal(x, tf.constant(threshold))
#     y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
#     return y


# def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
#     y_pred = threshold_binarize(y_pred, threshold)
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


# def dice_coef(y_true, y_pred, smooth=1.):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + K.epsilon()) / (
#                 K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())