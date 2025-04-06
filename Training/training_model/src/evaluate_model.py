import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc
)
import json


# Around line 16-20, keep the function signature
def evaluate_model(model, val_generator):
    """
    Evaluate the model and display results with comprehensive metrics
    Returns a dictionary of metrics for cross-validation analysis
    """
    # Get predictions
    # For generators, we need to get the exact number of samples
    if hasattr(val_generator, 'samples'):
        n_samples = val_generator.samples
    else:
        # For custom generators, use the length of diagnoses
        n_samples = len(val_generator.diagnoses) if hasattr(val_generator, 'diagnoses') else len(val_generator)
    
    # Reset the generator to ensure we start from the beginning
    if hasattr(val_generator, 'reset'):
        val_generator.reset()
    
    # Get predictions - make sure we get predictions for all samples
    try:
        # For tf.data.Dataset or generators that work with model.predict
        predictions = model.predict(val_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_proba = predictions  # Keep probabilities for AUC calculation
        
        # Get true labels
        if hasattr(val_generator, 'classes'):
            # For standard ImageDataGenerator
            y_true = val_generator.classes
        elif hasattr(val_generator, 'diagnoses') and hasattr(val_generator, 'diagnosis_to_idx'):
            # For custom generators
            y_true = np.array([val_generator.diagnosis_to_idx.get(d, 0) for d in val_generator.diagnoses])
        else:
            # For other generators, we need to extract labels from batches
            y_true = []
            for i in range(len(val_generator)):
                _, batch_labels = val_generator[i]
                y_true.extend(np.argmax(batch_labels, axis=1))
            y_true = np.array(y_true)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {}
    
    # Check for mismatch in sample counts
    if len(y_true) != len(y_pred):
        print(f"Warning: Mismatch in sample counts. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        # Adjust to the smaller size
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_pred_proba = y_pred_proba[:min_len]
        print(f"Adjusted to {min_len} samples for evaluation")
    
    # Get class names if available
    if hasattr(val_generator, 'diagnosis_to_idx'):
        idx_to_diagnosis = {v: k for k, v in val_generator.diagnosis_to_idx.items()}
        class_names = [idx_to_diagnosis.get(i, f"Class {i}") for i in range(len(val_generator.diagnosis_to_idx))]
    elif hasattr(val_generator, 'class_indices'):
        idx_to_class = {v: k for k, v in val_generator.class_indices.items()}
        class_names = [idx_to_class.get(i, f"Class {i}") for i in range(len(val_generator.class_indices))]
    else:
        # If class names are not available, use generic names
        num_classes = predictions.shape[1] if len(predictions.shape) > 1 else 2
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Calculate comprehensive metrics
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (macro-averaged)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Precision, Recall, F1 (weighted-averaged)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Sensitivity (same as recall) and Specificity (per class)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics
    n_classes = len(class_names)
    sensitivity_per_class = []
    specificity_per_class = []
    
    for i in range(n_classes):
        # True Positives, False Negatives, False Positives, True Negatives
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity_per_class.append(sensitivity)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    
    # Average sensitivity and specificity
    metrics['sensitivity_macro'] = np.mean(sensitivity_per_class)
    metrics['specificity_macro'] = np.mean(specificity_per_class)
    
    # ICBHI Score (average of sensitivity and specificity)
    metrics['icbhi_score'] = (metrics['sensitivity_macro'] + metrics['specificity_macro']) / 2
    
    # AUC calculation (one-vs-rest for multiclass)
    try:
        # Convert true labels to one-hot encoding for AUC calculation
        y_true_onehot = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_onehot[i, y_true[i]] = 1
        
        # Calculate AUC for each class
        auc_per_class = []
        for i in range(n_classes):
            if np.sum(y_true_onehot[:, i]) > 0:  # Only calculate if class exists in true labels
                auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
                auc_per_class.append(auc)
        
        # Average AUC across all classes
        metrics['auc_macro'] = np.mean(auc_per_class) if auc_per_class else 0
        
        # Weighted AUC
        metrics['auc_weighted'] = roc_auc_score(y_true_onehot, y_pred_proba, average='weighted', multi_class='ovr')
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        metrics['auc_macro'] = 0
        metrics['auc_weighted'] = 0
    
    # Get detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics['class_report'] = report
    
    # Print summary of metrics
    print("\nComprehensive Evaluation Metrics:")
    print("-" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall/Sensitivity (Macro): {metrics['recall_macro']:.4f}")
    print(f"Specificity (Macro): {metrics['specificity_macro']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"AUC (Macro): {metrics['auc_macro']:.4f}")
    print(f"ICBHI Score: {metrics['icbhi_score']:.4f}")
    print("-" * 60)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Return all metrics for further analysis
    return metrics


def save_model(model, save_path):
    """
    Save the trained model to disk
    """
    try:
        model.save(save_path)
        print(f"Model successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def save_metrics_to_json(metrics, file_path):
    """
    Save evaluation metrics to a JSON file
    
    Args:
        metrics: Dictionary of metrics
        file_path: Path to save the JSON file
    """
    # Convert numpy values to Python native types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if key == 'class_report':
            # Handle the classification report separately
            serializable_metrics[key] = {
                class_name: {
                    metric: float(val) if isinstance(val, (np.float32, np.float64)) else val
                    for metric, val in class_metrics.items()
                }
                for class_name, class_metrics in value.items()
            }
        elif isinstance(value, (np.float32, np.float64)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"Metrics saved to {file_path}")
