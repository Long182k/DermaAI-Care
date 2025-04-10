import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve

def calculate_class_weights(y_true, n_classes):
    """
    Calculate balanced class weights based on the frequency of each class in the dataset.
    More aggressive weighting for severely imbalanced datasets.
    
    Args:
        y_true: One-hot encoded class labels or class indices
        n_classes: Number of classes in the dataset
    
    Returns:
        Dictionary mapping class indices to weights
    """
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # One-hot encoded labels
        class_counts = np.sum(y_true, axis=0)
    else:
        # Class indices
        class_counts = np.bincount(y_true.astype(int).flatten(), minlength=n_classes)
    
    # Ensure we have counts for all classes
    if len(class_counts) < n_classes:
        class_counts = np.pad(class_counts, (0, n_classes - len(class_counts)), 'constant')
    
    # Calculate inverse frequency
    n_samples = np.sum(class_counts)
    weights = n_samples / (n_classes * class_counts)
    
    # Cap weights at 10.0 to prevent instability
    weights = np.minimum(weights, 10.0)
    
    # Normalize weights to have mean of 1.0
    weights = weights / np.mean(weights)
    
    return {i: weights[i] for i in range(n_classes)}

def calculate_adaptive_thresholds(y_true, y_pred_proba, class_weights=None, method='youden'):
    """
    Calculate optimal thresholds for each class using Youden's J statistic or PR curve.
    
    Args:
        y_true: One-hot encoded ground truth labels
        y_pred_proba: Predicted probabilities for each class
        class_weights: Optional dictionary mapping class indices to weights
        method: 'youden' for Youden's J statistic or 'pr' for PR curve
    
    Returns:
        Array of thresholds for each class
    """
    n_classes = y_pred_proba.shape[1]
    thresholds = np.zeros(n_classes)
    
    for i in range(n_classes):
        # Extract binary labels and predictions for this class
        if len(y_true.shape) > 1:
            y_true_i = y_true[:, i]
        else:
            y_true_i = (y_true == i).astype(int)
        
        y_pred_i = y_pred_proba[:, i]
        
        # Skip if no positive samples
        if np.sum(y_true_i) == 0:
            thresholds[i] = 0.5
            continue
        
        if method == 'youden':
            # Calculate ROC curve and find Youden's J statistic
            fpr, tpr, t = roc_curve(y_true_i, y_pred_i)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            thresholds[i] = t[best_idx]
        else:
            # PR curve for imbalanced classes
            precision, recall, t = precision_recall_curve(y_true_i, y_pred_i)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            thresholds[i] = t[best_idx]
    
    # If class weights are provided, adjust thresholds for rare classes
    if class_weights is not None:
        weight_array = np.array([class_weights[i] for i in range(n_classes)])
        # Lower threshold for classes with higher weights (rarer classes)
        scaling = 1.0 / np.sqrt(weight_array)
        scaling = scaling / np.mean(scaling)  # Normalize
        thresholds = thresholds * scaling
    
    # Ensure reasonable range
    thresholds = np.clip(thresholds, 0.1, 0.9)
    
    return thresholds

def apply_adaptive_thresholds(y_pred_proba, thresholds):
    """
    Apply class-specific thresholds to predicted probabilities.
    
    Args:
        y_pred_proba: Predicted probabilities for each class
        thresholds: Array of thresholds for each class
    
    Returns:
        Predicted class labels
    """
    n_samples = y_pred_proba.shape[0]
    n_classes = y_pred_proba.shape[1]
    
    # Initialize predictions
    y_pred = np.zeros((n_samples, n_classes), dtype=int)
    
    # Apply thresholds per class
    for i in range(n_classes):
        y_pred[:, i] = (y_pred_proba[:, i] >= thresholds[i]).astype(int)
    
    # For multi-class (not multi-label), ensure exactly one prediction per sample
    row_sums = np.sum(y_pred, axis=1)
    
    # If a sample has no predictions, assign it to the class with highest probability
    zero_rows = np.where(row_sums == 0)[0]
    if len(zero_rows) > 0:
        y_pred[zero_rows, np.argmax(y_pred_proba[zero_rows], axis=1)] = 1
    
    # If a sample has multiple predictions, keep only the highest probability one
    multi_rows = np.where(row_sums > 1)[0]
    if len(multi_rows) > 0:
        for row in multi_rows:
            # Reset row
            y_pred[row, :] = 0
            # Set highest probability class to 1
            y_pred[row, np.argmax(y_pred_proba[row])] = 1
    
    return y_pred

def correct_bias(model, class_weights, layer_name='predictions'):
    """
    Correct output layer bias to account for class imbalance.
    
    Args:
        model: Keras model to adjust
        class_weights: Dictionary of class weights
        layer_name: Name of the output layer
    
    Returns:
        Model with corrected biases
    """
    # Get original weights and biases
    for layer in model.layers:
        if layer.name == layer_name:
            weights, biases = layer.get_weights()
            n_classes = len(biases)
            
            # Calculate bias adjustments based on class weights
            weight_list = np.array([class_weights.get(i, 1.0) for i in range(n_classes)])
            log_class_weight = np.log(weight_list)
            
            # Adjust biases - reduced adjustment factor for stability
            adjustment_factor = 0.5  # Use 0.5 instead of 1.0 to be more conservative
            new_biases = biases + (log_class_weight * adjustment_factor)
            
            # Set new weights
            layer.set_weights([weights, new_biases])
            
            print(f"Applied bias correction: Original biases = {biases}")
            print(f"New biases = {new_biases}")
            
            break
    
    return model

def detect_prediction_bias(model, validation_data, validation_labels, threshold=0.9):
    """
    Detect if the model is biased toward predicting certain classes.
    
    Args:
        model: Keras model to check
        validation_data: Validation dataset features
        validation_labels: Ground truth labels (one-hot encoded)
        threshold: Threshold for detecting bias
    
    Returns:
        Tuple of (is_biased, dominant_class) where is_biased is a boolean and
        dominant_class is the index of the dominating class or -1 if none
    """
    # Get predictions
    predictions = model.predict(validation_data)
    
    # Convert to class labels
    if predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = (predictions > 0.5).astype(int).flatten()
    
    # Count class occurrences
    n_classes = validation_labels.shape[1] if len(validation_labels.shape) > 1 else np.max(validation_labels) + 1
    class_counts = np.bincount(pred_classes, minlength=n_classes)
    class_proportions = class_counts / np.sum(class_counts)
    
    # Check if any class dominates predictions
    max_proportion = np.max(class_proportions)
    dominant_class = np.argmax(class_proportions)
    
    is_biased = max_proportion > threshold
    
    if is_biased:
        print(f"WARNING: Model shows prediction bias. Class {dominant_class} accounts for {max_proportion:.1%} of all predictions.")
        print(f"Class distribution in predictions: {class_proportions}")
    
    return is_biased, dominant_class 