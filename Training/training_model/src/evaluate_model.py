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
def evaluate_model(model, test_generator, class_names, output_dir=None):
    """
    Evaluate the model and generate metrics, confusion matrix, and classification report.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator or dataset (can be YOLODetectionGenerator or tf.data.Dataset)
        class_names: List of class names
        output_dir: Directory to save evaluation results
    
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
                
        # Convert predictions to class indices
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Convert true labels to class indices if they're one-hot encoded
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_classes = np.argmax(y_true, axis=1)
        else:
            y_true_classes = y_true.astype(int)
        
        # Generate classification report
        # Get unique classes in the predicted and true labels
        unique_classes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
        
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
                output_dict=True
            )
        except ValueError as e:
            print(f"Error with classification report: {e}")
            # Fallback approach - generate report without target_names
            report = classification_report(
                y_true_classes,
                y_pred_classes,
                labels=used_class_indices,
                output_dict=True
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
        
        print("\n" + "-" * 50)
        # Print accuracy
        print(f"{'accuracy':>12} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['macro avg']['support']:>10.0f}")
        
        # Print macro and weighted averages
        print(f"{'macro avg':>12} {report['macro avg']['precision']:>10.2f} {report['macro avg']['recall']:>10.2f} {report['macro avg']['f1-score']:>10.2f} {report['macro avg']['support']:>10.0f}")
        print(f"{'weighted avg':>12} {report['weighted avg']['precision']:>10.2f} {report['weighted avg']['recall']:>10.2f} {report['weighted avg']['f1-score']:>10.2f} {report['weighted avg']['support']:>10.0f}")
        
        # Calculate metrics from the classification report for consistency
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        # Calculate AUC for binary classification
        if len(used_class_names) == 2:
            try:
                # For binary classification
                if y_pred.shape[1] >= 2:
                    auc = roc_auc_score(y_true_classes, y_pred[:, 1])
                else:
                    # Single output neuron case
                    auc = roc_auc_score(y_true_classes, y_pred.flatten())
            except Exception as e:
                print(f"Error calculating binary AUC: {e}")
                # Fallback if error with AUC calculation
                auc = 0.5
        else:
            try:
                # Ensure y_pred has the right shape for multi-class ROC AUC
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # For multi-class classification, ensure we're only using classes present in data
                    # Check if we have enough columns in y_pred for all classes
                    if y_pred.shape[1] >= max(used_class_indices) + 1:
                        # Extract only the probabilities for classes we have
                        y_pred_subset = np.zeros((y_pred.shape[0], len(used_class_indices)))
                        for i, idx in enumerate(used_class_indices):
                            if idx < y_pred.shape[1]:
                                y_pred_subset[:, i] = y_pred[:, idx]
                        
                        # Use the subset for AUC calculation
                        auc = roc_auc_score(
                            tf.keras.utils.to_categorical(y_true_classes, num_classes=len(used_class_indices)), 
                            y_pred_subset, 
                            multi_class='ovr'
                        )
                    else:
                        print(f"Warning: y_pred shape {y_pred.shape} doesn't match required classes {used_class_indices}")
                        # Fall back to calculating AUC with available classes
                        auc = roc_auc_score(y_true_classes, y_pred, multi_class='ovr')
                else:
                    # If predictions are single-dimensional, convert to probabilities
                    auc = roc_auc_score(y_true_classes, tf.keras.utils.to_categorical(y_pred_classes), multi_class='ovr')
            except Exception as e:
                print(f"Error calculating multi-class AUC: {e}")
                # Fallback if error with AUC calculation
                auc = 0.5
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes, labels=used_class_indices)
        
        # Calculate specificity for binary classification
        if len(used_class_names) == 2:
            tn = cm[0, 0]
            fp = cm[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # For multi-class, calculate average specificity
            specificities = []
            for i in range(len(used_class_indices)):
                # True negatives are all the samples that are not in class i and not predicted as class i
                tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
                # False positives are samples that are not in class i but predicted as class i
                fp = np.sum(np.delete(cm[:, i], i))
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificities.append(spec)
            specificity = np.mean(specificities)
        
        # Calculate ICBHI score (average of sensitivity and specificity)
        icbhi_score = (recall + specificity) / 2
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save classification report
        report_path = os.path.join(output_dir, f"classification_report_{timestamp}.csv")
        pd.DataFrame(report).to_csv(report_path)
        print(f"Classification report saved to: {report_path}")
        
        # Save confusion matrix visualization
        cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        plot_confusion_matrix(y_true_classes, y_pred_classes, used_class_names, cm_path, used_class_indices)
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Print metrics summary
        print("\nEvaluation Metrics:")
        print("-" * 55)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall/Sensitivity: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Specificity: {specificity:.4f}")
        print(f"Validation ICBHI Score: {icbhi_score:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'icbhi_score': icbhi_score,
            'confusion_matrix_path': cm_path,
            'report_path': report_path
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
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
