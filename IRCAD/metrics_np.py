import numpy as np

epsilon = np.finfo(np.float).eps #2.2E-16
def np_metrics(TotalClasses, y_true, y_pred, method="image", verbose=False):
    def metric_mean(metric, TotalClassNumber):
        if TotalClassNumber>1:
            metric = metric/TotalClassNumber
        else:
            metric = metric
        return metric
    
    split_accuracy = 0.0
    split_jaccard = 0.0
    split_dice = 0.0
    split_recall = 0.0
    split_precision = 0.0
    avg_split_metrics = {"avg_accuracy":[],"avg_jaccard":[],"avg_dice":[],"avg_recall":[],"avg_precision":[]} # defaultdict(list)
    
    class_preds = np.argmax(y_pred,axis=-1)
    class_metrics = {"accuracy":[],"jaccard":[],"dice":[],"recall":[],"precision":[]} # defaultdict(list)
    
    DesiredClasses = list(range(TotalClasses)) #TotalClasses includes background

    for TargetClass in DesiredClasses:
        if verbose: print("Target class {}".format(TargetClass))

        ########################################################
        # Identify locations where class exists and where class has been identified by model.predict
        ########################################################
        predicted_mask = np.equal(class_preds, TargetClass, dtype=object)#np.float32) #TF version performed both elementwise matching and conversion from tensor to float32 but this returns True/False?
        inv_predicted_mask = np.not_equal(class_preds, TargetClass, dtype=object)
        actual_mask = np.equal(y_true[:,:,:,TargetClass], 1, dtype=object)
        if verbose: print(y_true[:,:,:,TargetClass])
        inv_actual_mask = np.not_equal(y_true[:,:,:,TargetClass], 1, dtype=object)
        if verbose: print("Actual mask is {}, Predicted mask is {}".format(actual_mask, predicted_mask))
        ########################################################
        # Sum across full image and then calculate metrics
        ########################################################
        #Calculate confusion matrix info - to be returned if required in future
        tp = np.sum(actual_mask * predicted_mask, axis=(0,1,2))
        fp = np.sum(predicted_mask, axis=(0,1,2)) - tp
        fn = np.sum(actual_mask, axis=(0,1,2)) - tp
        tn = np.sum(inv_predicted_mask, axis=(0,1,2)) - fn
        print("True positive {} False positive {} False negative {} True negative {}".format(tp, fp, fn, tn))
        
        if method == "image":
            #Get the intersection and union
            total_class_intersection = np.sum(predicted_mask*actual_mask, axis=(0,1,2)) #?Remove axis 0 
            total_class_sum = np.sum(actual_mask + predicted_mask, axis=(0,1,2))
            #Calculate metrics for each class
            class_accuracy = (tp + tn)/(tp + tn + fp + fn)
            #print("accuracy  is: {}".format(accuracy))
            class_jaccard = (total_class_intersection + epsilon) / (total_class_sum - total_class_intersection + epsilon)
            class_dice = ((2. * total_class_intersection) + epsilon) / (total_class_sum + epsilon)
            class_recall = (tp + epsilon) / (tp+fn + epsilon) #epsilon (10^-16) added to avoid div0 errors
            class_precision = (tp + epsilon) / (tp+fp + epsilon)
        
        elif method == "pixel":
            class_intersection = np.sum(predicted_mask*actual_mask, axis=(0,1,2)) #np.sum(np.logical_and(actual_mask, predicted_mask)) ### Still need to change from (summation over all images then metrics) to (metrics per image then summation) ###
            class_sum = np.sum(actual_mask, axis=(0,1,2)) + np.sum(predicted_mask, axis=(0,1,2)) #actual_mask.sum() + predicted_mask.sum() #
            if verbose: print("Class intersection is {} Class sum is {}".format(class_intersection, class_sum))
            class_accuracy = (tp + tn)/(tp + tn + fp + fn)
            class_jaccard = (class_intersection + epsilon)/(class_sum-class_intersection + epsilon)
            class_dice = (2.*class_intersection + epsilon)/(class_sum + epsilon)
            class_recall = (tp + epsilon) / (tp+fn + epsilon) #epsilon (10^-16) added to avoid div0 errors
            class_precision = (tp + epsilon) / (tp+fp + epsilon)            
        
        print("Class Metrics for class {} - accuracy: {} jaccard: {} dice: {} recall: {} precision: {}".format(TargetClass, class_accuracy, class_jaccard, class_dice, class_recall, class_precision))
        
        ###Definitely better way to do this than iteratively construct and iteratively deconstruct into plots###
        
        class_metrics["accuracy"].append(class_accuracy)    
        class_metrics["jaccard"].append(class_jaccard)
        class_metrics["dice"].append(class_dice)
        class_metrics["recall"].append(class_recall)
        class_metrics["precision"].append(class_precision)

        split_accuracy += class_accuracy
        split_jaccard += class_jaccard
        split_dice += class_dice
        split_recall += class_recall
        split_precision += class_precision

    avg_split_accuracy = metric_mean(split_accuracy, TotalClasses)
    avg_split_jaccard = metric_mean(split_jaccard, TotalClasses)
    avg_split_dice = metric_mean(split_dice, TotalClasses)
    avg_split_recall = metric_mean(split_recall, TotalClasses)
    avg_split_precision = metric_mean(split_precision, TotalClasses)

    avg_split_metrics["avg_accuracy"].append(avg_split_accuracy)    
    avg_split_metrics["avg_jaccard"].append(avg_split_jaccard)
    avg_split_metrics["avg_dice"].append(avg_split_dice)
    avg_split_metrics["avg_recall"].append(avg_split_recall)
    avg_split_metrics["avg_precision"].append(avg_split_precision)
    
    print("Mean of metrics this split - accuracy: {} jaccard: {} dice: {} recall: {} precision: {}".format(avg_split_accuracy, avg_split_jaccard, avg_split_dice, avg_split_recall, avg_split_precision))

               
    return class_metrics