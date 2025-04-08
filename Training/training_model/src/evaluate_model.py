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
        test_generator: Test data generator (YOLODetectionGenerator)
        class_names: List of class names
        output_dir: Directory to save evaluation results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(model_save_path), "evaluation_results")
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating predictions...")
        
        # Get predictions and true labels
        y_pred = []
        y_true = []
        
        # Process batches
        for i in range(len(test_generator)):
            batch_data, batch_labels = test_generator[i]
            batch_pred = model.predict(batch_data, verbose=0)
            y_pred.extend(batch_pred)
            y_true.extend(batch_labels)
        
        # Convert to numpy arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
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
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Print classification report in a tabular format
        print("\nClassification Report:")
        print("-" * 50)
        print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        print("-" * 50)
        
        # Print metrics for each class
        for i, class_name in enumerate(class_names):
            metrics = report[class_name]
            print(f"{i:>12} {metrics['precision']:>10.2f} {metrics['recall']:>10.2f} {metrics['f1-score']:>10.2f} {metrics['support']:>10.0f}")
        
        print("\n" + "-" * 50)
        # Print accuracy
        print(f"{'accuracy':>12} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['macro avg']['support']:>10.0f}")
        
        # Print macro and weighted averages
        print(f"{'macro avg':>12} {report['macro avg']['precision']:>10.2f} {report['macro avg']['recall']:>10.2f} {report['macro avg']['f1-score']:>10.2f} {report['macro avg']['support']:>10.0f}")
        print(f"{'weighted avg':>12} {report['weighted avg']['precision']:>10.2f} {report['weighted avg']['recall']:>10.2f} {report['weighted avg']['f1-score']:>10.2f} {report['weighted avg']['support']:>10.0f}")
        
        # Calculate additional metrics using scikit-learn functions
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        
        # Calculate AUC if it's a binary classification
        if len(class_names) == 2:
            auc_score = roc_auc_score(y_true_classes, y_pred[:, 1])
        else:
            auc_score = roc_auc_score(y_true_classes, y_pred, multi_class='ovr')
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()
        specificity = tn / (tn + fp)
        
        # Calculate ICBHI score (average of sensitivity and specificity)
        icbhi_score = (recall + specificity) / 2
        
        # Save classification report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"classification_report_{timestamp}.csv")
        pd.DataFrame(report).to_csv(report_path)
        print(f"\nClassification report saved to: {report_path}")
        
        # Generate and save confusion matrix
        cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, output_dir)
        print(f"Confusion matrix saved to: {cm_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
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

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path)
    plt.close()
    
    return cm_path
