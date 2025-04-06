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
    Evaluate the model and display results with enhanced AUC calculation
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
        predictions = predictions[:min_len]
        print(f"Adjusted to {min_len} samples for evaluation")
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    
    # Get class names if available
    if hasattr(val_generator, 'diagnosis_to_idx'):
        idx_to_class = {v: k for k, v in val_generator.diagnosis_to_idx.items()}
        target_names = [idx_to_class.get(i, f"Class {i}") for i in range(len(val_generator.diagnosis_to_idx))]
    else:
        target_names = [f"Class {i}" for i in range(np.max(y_true) + 1)]
    
    # Print classification report
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Calculate and print macro-averaged metrics (better for imbalanced classes)
    print("\nMacro-averaged Metrics (better for imbalanced classes):")
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Macro Precision: {macro_precision:.2%}")
    print(f"Macro Recall: {macro_recall:.2%}")
    print(f"Macro F1 Score: {macro_f1:.2%}")
    
    # Calculate validation metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate weighted metrics
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate AUC for multi-class
    # One-hot encode the true labels for ROC calculation
    y_true_onehot = np.zeros((len(y_true), len(np.unique(y_true))))
    for i, val in enumerate(y_true):
        y_true_onehot[i, val] = 1
    
    # Calculate AUC
    try:
        auc_value = roc_auc_score(y_true_onehot, predictions, multi_class='ovr')
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc_value = 0
    
    # Calculate specificity (true negative rate)
    # For multi-class, we calculate specificity for each class and average
    specificities = []
    for cls in range(len(np.unique(y_true))):
        # Create binary classification for this class
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    
    # Average specificity
    specificity = np.mean(specificities)
    
    # Calculate ICBHI score (harmonic mean of sensitivity and specificity)
    sensitivity = macro_recall  # Sensitivity is the same as recall
    icbhi_score = 2 * sensitivity * specificity / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0
    
    # Print validation metrics
    print("\nValidation Metrics:")
    print(f"Validation Accuracy: {accuracy:.2%}")
    print(f"Validation Precision: {weighted_precision:.2%}")
    print(f"Validation Recall/Sensitivity: {sensitivity:.2%}")
    print(f"Validation F1 Score: {weighted_f1:.2%}")
    print(f"Validation AUC: {auc_value:.2%}")
    print(f"Validation Specificity: {specificity:.2%}")
    print(f"Validation ICBHI Score: {icbhi_score:.2%}")
    
    # Return metrics for cross-validation analysis
    return {
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': sensitivity,
        'f1': weighted_f1,
        'auc': auc_value,
        'specificity': specificity,
        'icbhi_score': icbhi_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'class_report': report
    }


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
