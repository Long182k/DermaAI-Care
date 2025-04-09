import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
import tensorflow as tf
import os
import pandas as pd
from datetime import datetime
import traceback
import gc


# Around line 16-20, keep the function signature
def evaluate_model(model, test_generator, class_names, output_dir=None, prediction_thresholds=None):
    """
    Evaluate the model and generate metrics, confusion matrix, and classification report.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator or dataset (can be YOLODetectionGenerator or tf.data.Dataset)
        class_names: List of class names
        output_dir: Directory to save evaluation results
        prediction_thresholds: Dictionary mapping class indices to custom threshold values
    
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = "/kaggle/working/DermaAI-Care/Training/training_model/models/evaluation_results"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate predictions
        print("Generating predictions...")
        
        # Handle different types of generators
        if isinstance(test_generator, tf.data.Dataset):
            # For tf.data.Dataset
            y_true = []
            y_pred = []
            
            # Process batches
            for images, labels in test_generator:
                batch_pred = model.predict(images, verbose=0)
                y_pred.extend(batch_pred)
                y_true.extend(labels.numpy())
                
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
        else:
            # For custom generators
            y_true = []
            y_pred = []
            
            # For custom generators
            for i in range(len(test_generator)):
                try:
                    batch_data, batch_labels = test_generator[i]
                    batch_pred = model.predict(batch_data, verbose=0)
                    y_pred.extend(batch_pred)
                    y_true.extend(batch_labels)
                except Exception as batch_error:
                    print(f"Error processing batch {i}: {batch_error}")
                    continue
            
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

        # Force garbage collection to free up memory
        gc.collect()
        
        # Debug information
        print(f"Predictions shape: {y_pred.shape}, True labels shape: {y_true.shape}")
        
        # Apply custom thresholds for each class if provided
        if prediction_thresholds and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            print(f"Applying custom prediction thresholds: {prediction_thresholds}")
            
            # Initialize prediction classes array with zeros
            y_pred_classes = np.zeros(y_pred.shape[0], dtype=int)
            
            # For each sample, check each class according to its threshold
            for sample_idx in range(y_pred.shape[0]):
                sample_pred = y_pred[sample_idx]
                
                # Apply different thresholds for each class
                # We'll pick the class with highest confidence above its threshold
                max_conf = -1
                max_class = -1
                
                for class_idx in range(y_pred.shape[1]):
                    # Get the confidence for this class
                    conf = sample_pred[class_idx]
                    
                    # Get appropriate threshold (default to 0.5 if not specified)
                    threshold = 0.5
                    if str(class_idx) in prediction_thresholds:
                        threshold = prediction_thresholds[str(class_idx)]
                    elif class_idx in prediction_thresholds:
                        threshold = prediction_thresholds[class_idx]
                    
                    # Check if this prediction passes the threshold and has highest confidence
                    if conf > threshold and conf > max_conf:
                        max_conf = conf
                        max_class = class_idx
                
                # If no class passed its threshold, use the original argmax
                if max_class == -1:
                    # Use a lower minimum threshold (0.1) just to avoid no predictions at all
                    max_class = np.argmax(sample_pred)
                    if sample_pred[max_class] > 0.1:
                        y_pred_classes[sample_idx] = max_class
                    else:
                        # If all predictions are below 0.1, use a fallback:
                        # Pick the most common class in training data (typically class 1 for skin data)
                        y_pred_classes[sample_idx] = 1  # NV is usually most common
                else:
                    y_pred_classes[sample_idx] = max_class
        elif len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Standard approach - take argmax for multi-class
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            # Binary case
            threshold = 0.5
            if prediction_thresholds and 0 in prediction_thresholds:
                threshold = prediction_thresholds[0]
            y_pred_classes = (y_pred > threshold).astype(int)
        
        # Convert true labels to class indices if they're one-hot encoded
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_classes = np.argmax(y_true, axis=1)
        else:
            y_true_classes = y_true.astype(int)
        
        # Generate classification report
        # Get unique classes from the training data (not just predicted classes)
        unique_classes = sorted(np.unique(y_true_classes))
        
        # Map the unique class indices to class names
        used_class_names = []
        used_class_indices = []
        for idx in unique_classes:
            if idx < len(class_names):
                used_class_names.append(class_names[idx])
                used_class_indices.append(idx)
            else:
                # Handle case where index is out of bounds
                used_class_names.append(f"Class_{idx}")
                used_class_indices.append(idx)
        
        print(f"Generating classification report for {len(used_class_names)} classes: {used_class_names}")
        
        # Generate the report with the labels parameter to specify which classes to include
        try:
            report = classification_report(
                y_true_classes,
                y_pred_classes,
                labels=used_class_indices,
                target_names=used_class_names,
                output_dict=True,
                zero_division=0  # Handle zero division explicitly
            )
        except ValueError as e:
            print(f"Error with classification report: {e}")
            # Fallback approach - generate report without target_names
            report = classification_report(
                y_true_classes,
                y_pred_classes,
                labels=used_class_indices,
                output_dict=True,
                zero_division=0  # Handle zero division explicitly
            )
            
            # Add class names manually
            new_report = {}
            for idx, label in enumerate(used_class_indices):
                if str(label) in report:
                    if idx < len(used_class_names):
                        new_report[used_class_names[idx]] = report[str(label)]
                    else:
                        new_report[f"Class_{label}"] = report[str(label)]
            
            # Copy other metrics
            for key in ['accuracy', 'macro avg', 'weighted avg']:
                if key in report:
                    new_report[key] = report[key]
            
            report = new_report
        
        # Add debugging information to understand what the model is predicting
        print("\nPrediction Distribution Analysis:")
        print("-" * 50)
        
        # Count predictions per class
        classes_to_count = max(len(used_class_names), np.max(y_pred_classes) + 1)
        pred_counts = np.bincount(y_pred_classes, minlength=classes_to_count)
        print("Predictions per class:")
        for i, count in enumerate(pred_counts):
            if i < len(class_names):
                class_name = class_names[i]
                total = np.sum(y_true_classes == i)
                percentage = count/max(1, total)*100
                print(f"  Class {i} ({class_name}): {count} predictions out of {total} samples ({percentage:.2f}%)")
        
        # Print raw prediction distribution
        print("\nPrediction confidence distribution:")
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # For multi-class, show average confidence per class
            mean_confidences = np.mean(y_pred, axis=0)
            max_confidences = np.max(y_pred, axis=0)
            
            for i, (mean_conf, max_conf) in enumerate(zip(mean_confidences, max_confidences)):
                if i < len(class_names):
                    class_name = class_names[i]
                    print(f"  Class {i} ({class_name}): Avg confidence: {mean_conf:.4f}, Max confidence: {max_conf:.4f}")
                    
            # Show histogram of highest confidence predictions
            top_class_indices = np.argmax(y_pred, axis=1)
            top_class_confidences = np.max(y_pred, axis=1)
            
            print("\nDistribution of highest confidence predictions:")
            for i in range(len(class_names)):
                class_name = class_names[i]
                class_mask = top_class_indices == i
                count = np.sum(class_mask)
                if count > 0:
                    avg_conf = np.mean(top_class_confidences[class_mask])
                    print(f"  Class {i} ({class_name}): {count} samples with avg confidence {avg_conf:.4f}")
                else:
                    print(f"  Class {i} ({class_name}): 0 samples")
        
        # Print classification report in a tabular format
        print("\nClassification Report:")
        print("-" * 50)
        print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        print("-" * 50)
        
        # Print metrics for each class
        for i, class_name in enumerate(used_class_names):
            if class_name in report:
                metrics = report[class_name]
                print(f"{i:>12} {metrics['precision']:>10.2f} {metrics['recall']:>10.2f} {metrics['f1-score']:>10.2f} {metrics['support']:>10.0f}")
            else:
                print(f"{i:>12} {0:>10.2f} {0:>10.2f} {0:>10.2f} {0:>10.0f}")
        
        print("-" * 50)
        if 'accuracy' in report:
            print(f"{'accuracy':>12} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['macro avg']['support']:>10.0f}")
        if 'macro avg' in report:
            print(f"{'macro avg':>12} {report['macro avg']['precision']:>10.2f} {report['macro avg']['recall']:>10.2f} {report['macro avg']['f1-score']:>10.2f} {report['macro avg']['support']:>10.0f}")
        if 'weighted avg' in report:
            print(f"{'weighted avg':>12} {report['weighted avg']['precision']:>10.2f} {report['weighted avg']['recall']:>10.2f} {report['weighted avg']['f1-score']:>10.2f} {report['weighted avg']['support']:>10.0f}")

        # Save classification report to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"classification_report_{timestamp}.csv")
        
        # Convert report to DataFrame for easy saving
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(report_path)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes, labels=used_class_indices)
        
        # Plot confusion matrix
        confusion_matrix_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        plot_confusion_matrix(y_true_classes, y_pred_classes, used_class_names, confusion_matrix_path, used_class_indices)
        
        # Additional evaluation: AUC and specificity
        # Calculate AUC only if there are multiple classes
        auc_value = 0
        try:
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Multi-class or multi-label
                if y_pred.shape[1] == 2:
                    # Binary problem represented as 2 outputs
                    auc_value = roc_auc_score(y_true[:, 1], y_pred[:, 1])
                else:
                    # True multi-class problem
                    if np.max(y_true_classes) < y_pred.shape[1]:
                        # Check if we have enough classes in the true labels
                        auc_value = roc_auc_score(
                            tf.keras.utils.to_categorical(y_true_classes, num_classes=y_pred.shape[1]), 
                            y_pred, 
                            multi_class='ovr', 
                            average='weighted'
                        )
                    else:
                        print("Warning: Class mismatch between predictions and true labels. Using fallback AUC calculation.")
                        # Fallback AUC calculation
                        auc_value = 0.5  # Default value
            else:
                # Binary problem with single output
                auc_value = roc_auc_score(y_true, y_pred)
        except Exception as auc_error:
            print(f"Error calculating AUC: {auc_error}")
            auc_value = 0.5  # Default fallback
            
        # Calculate sensitivity (recall) and specificity
        # Sensitivity is already calculated as recall
        sensitivity = report['macro avg']['recall']
        
        # Specificity calculation
        specificity = 0
        spec_scores = []
        
        # Calculate specificity for each class
        for i in range(len(used_class_indices)):
            true_negatives = np.sum((y_true_classes != used_class_indices[i]) & (y_pred_classes != used_class_indices[i]))
            false_positives = np.sum((y_true_classes != used_class_indices[i]) & (y_pred_classes == used_class_indices[i]))
            
            if true_negatives + false_positives > 0:
                class_specificity = true_negatives / (true_negatives + false_positives)
                spec_scores.append(class_specificity)
            else:
                spec_scores.append(0)
        
        # Macro-average specificity
        specificity = np.mean(spec_scores) if spec_scores else 0
        
        # Calculate ICBHI score (average of sensitivity and specificity)
        icbhi_score = (sensitivity + specificity) / 2
        
        # Compile all metrics
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score'],
            'auc': auc_value,
            'specificity': specificity,
            'icbhi_score': icbhi_score,
            'confusion_matrix_path': confusion_matrix_path,
            'report_path': report_path
        }
        
        # Print ICBHI score components
        print(f"Validation ICBHI Score: {icbhi_score:.4f}")
        print(f"  - Sensitivity: {sensitivity:.4f}")
        print(f"  - Specificity: {specificity:.4f}")
        
        # Return metrics
        return metrics
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()
        return None

def evaluate_multilabel(model, test_generator, predictions, class_names, output_dir, timestamp):
    """
    Evaluate multi-label classification model
    """
    # Get true labels
    true_labels = np.array([])
    for i in range(len(test_generator)):
        _, batch_labels = test_generator[i]
        if len(true_labels) == 0:
            true_labels = batch_labels
        else:
            true_labels = np.vstack((true_labels, batch_labels))
    
    # Calculate metrics for each class
    metrics = {}
    for i, class_name in enumerate(class_names):
        # Calculate metrics
        y_true = true_labels[:, i]
        y_pred = predictions[:, i]
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Store metrics
        metrics[class_name] = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
    
    if output_dir:
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for class_name in class_names:
            plt.plot(
                metrics[class_name]['fpr'],
                metrics[class_name]['tpr'],
                label=f'{class_name} (AUC = {metrics[class_name]["auc"]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'roc_curves_multilabel_{timestamp}.png'))
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'AUC': [metrics[c]['auc'] for c in class_names]
        })
        metrics_df.to_csv(os.path.join(output_dir, f'evaluation_metrics_{timestamp}.csv'), index=False)
        
        # Generate confusion matrices
        for i, class_name in enumerate(class_names):
            y_true = true_labels[:, i]
            y_pred = (predictions[:, i] > 0.5).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {class_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{class_name}_{timestamp}.png'))
            plt.close()
    
    return metrics

def evaluate_singlelabel(model, test_generator, predictions, class_names, output_dir, timestamp):
    """
    Evaluate single-label classification model
    """
    # Get predicted classes
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = []
    for i in range(len(test_generator)):
        _, batch_labels = test_generator[i]
        y_true.extend(np.argmax(batch_labels, axis=1))
    y_true = np.array(y_true)
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, f'classification_report_{timestamp}.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to: {report_path}")
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'confusion_matrix_path': cm_path,
        'report_path': report_path
    }

def plot_training_history(history, output_dir=None):
    """
    Plot training history metrics
    
    Args:
        history: Keras history object
        output_dir: Directory to save plots
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot training curves
        metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
        for metric in metrics:
            if metric in history.history:
                plt.figure(figsize=(10, 6))
                plt.plot(history.history[metric], label=f'Training {metric}')
                if f'val_{metric}' in history.history:
                    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                plt.title(f'Model {metric}')
                plt.xlabel('Epoch')
                plt.ylabel(metric.capitalize())
                plt.legend()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'{metric}_history_{timestamp}.png'))
                plt.close()
        
        # Save history to CSV
        if output_dir:
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(os.path.join(output_dir, f'training_history_{timestamp}.csv'))
            
    except Exception as e:
        print(f"Error plotting training history: {e}")

def save_model(model, save_path):
    """
    Save the trained model to disk
    """
    try:
        model.save(save_path)
        print(f"Model successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_path, class_indices=None):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
        class_indices: Optional list of class indices to include in the matrix
    """
    try:
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_indices)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        traceback.print_exc()
