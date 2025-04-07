import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score as sk_roc_auc_score,
    f1_score as sk_f1_score,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score
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
    Evaluate the model and generate comprehensive performance metrics
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: List of class names
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get predictions
        print("Generating predictions...")
        predictions = model.predict(test_generator)
        
        # For multi-label classification
        if isinstance(predictions, list) or (isinstance(predictions, np.ndarray) and len(predictions.shape) > 2):
            return evaluate_multilabel(model, test_generator, predictions, class_names, output_dir, timestamp)
        
        # For single-label classification
        return evaluate_singlelabel(model, test_generator, predictions, class_names, output_dir, timestamp)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
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
    
    if output_dir:
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{timestamp}.png'))
        plt.close()
        
        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, f'classification_report_{timestamp}.csv'))
        
        # Calculate and plot ROC curves
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            y_pred_proba = predictions[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, f'roc_curves_{timestamp}.png'))
        plt.close()
    
    return {
        'classification_report': report,
        'confusion_matrix': cm
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

# Helper functions for multi-label metrics
def precision_score(y_true, y_pred):
    """Calculate precision for binary predictions"""
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall_score(y_true, y_pred):
    """Calculate recall for binary predictions"""
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1_score(y_true, y_pred):
    """Calculate F1 score for binary predictions"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

def roc_auc_score(y_true, y_score):
    """Calculate ROC AUC score"""
    try:
        if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
            # If all samples are of the same class, ROC AUC is undefined
            return 0.5
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return auc(fpr, tpr)
    except:
        return 0.5
