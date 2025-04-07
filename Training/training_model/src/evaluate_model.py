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
def evaluate_model(model, val_generator, class_indices=None, save_path=None):
    """
    Comprehensive model evaluation with nicely formatted metrics display
    
    Args:
        model: Trained Keras model
        val_generator: Validation data generator
        class_indices: Dictionary mapping indices to class names
        save_path: Path to save evaluation results (default: models directory)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    if save_path is None:
        save_path = "/kaggle/working/DermaAI-Care/Training/training_model/models"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get current timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Reverse class_indices dictionary to get index to class name mapping
    if class_indices is not None:
        idx_to_class = {v: k for k, v in class_indices.items()}
    else:
        # If no class names provided, use indices
        n_classes = val_generator.num_classes if hasattr(val_generator, 'num_classes') else model.output_shape[-1]
        idx_to_class = {i: f"Class {i}" for i in range(n_classes)}
    
    # Determine if this is a multi-label classification problem
    multi_label = True if model.output_shape[-1] > 1 and model.layers[-1].activation.__name__ == 'sigmoid' else False
    
    try:
        # Get predictions and true labels
        print("Generating predictions for evaluation...")
        all_labels = []
        all_preds = []
        all_pred_probs = []
        
        # Predict on batches
        for i in range(len(val_generator)):
            X_batch, y_batch = val_generator[i]
            pred_probs = model.predict(X_batch, verbose=0)
            
            # For multi-label classification, use 0.5 threshold
            if multi_label:
                preds = (pred_probs >= 0.5).astype(int)
            else:
                preds = np.argmax(pred_probs, axis=1)
                y_batch = np.argmax(y_batch, axis=1)
            
            all_labels.append(y_batch)
            all_preds.append(preds)
            all_pred_probs.append(pred_probs)
        
        # Combine batches
        if multi_label:
            y_true = np.vstack(all_labels)
            y_pred = np.vstack(all_preds)
            y_pred_probs = np.vstack(all_pred_probs)
        else:
            y_true = np.concatenate(all_labels)
            if len(np.array(all_preds).shape) > 1 and np.array(all_preds).shape[1] > 1:
                y_pred = np.concatenate([np.argmax(p, axis=1) for p in all_preds])
            else:
                y_pred = np.concatenate(all_preds)
            y_pred_probs = np.vstack(all_pred_probs)
        
        # Match dimensions if needed
        if y_true.shape[0] != y_pred.shape[0]:
            print(f"Warning: Shape mismatch between true ({y_true.shape}) and predicted ({y_pred.shape})")
            min_samples = min(y_true.shape[0], y_pred.shape[0])
            y_true = y_true[:min_samples]
            y_pred = y_pred[:min_samples]
            y_pred_probs = y_pred_probs[:min_samples]
        
        print(f"Evaluating on {y_true.shape[0]} samples")
        
        # Generate metrics based on the type of classification problem
        metrics = {}
        
        # Calculate confusion matrix
        if multi_label:
            # For multi-label, calculate confusion matrix for each class
            cms = []
            for i in range(y_true.shape[1]):
                cm = confusion_matrix(y_true[:, i], y_pred[:, i])
                if cm.shape[0] < 2 or cm.shape[1] < 2:
                    # Add missing rows/columns if needed
                    if cm.shape[0] < 2 and cm.shape[1] < 2:
                        # Only one class predicted, add a row and column of zeros
                        if cm[0][0] == 0:  # All negatives
                            cm = np.array([[cm[0][0], 0], [0, 0]])
                        else:  # All positives
                            cm = np.array([[0, 0], [0, cm[0][0]]])
                    elif cm.shape[0] < 2:
                        # Add a row of zeros
                        cm = np.vstack([np.zeros((1, cm.shape[1])), cm])
                    elif cm.shape[1] < 2:
                        # Add a column of zeros
                        cm = np.hstack([np.zeros((cm.shape[0], 1)), cm])
                cms.append(cm)
        else:
            cm = confusion_matrix(y_true, y_pred)
        
        # Calculate and display classification report with nice formatting
        print("\nClassification Report:")
        print("-" * 55)
        
        # Generate comprehensive classification report
        if multi_label:
            # For multi-label, calculate metrics for each class
            class_metrics = {}
            for i in range(y_true.shape[1]):
                class_name = idx_to_class[i]
                
                # Skip if no samples in this class
                if np.sum(y_true[:, i]) == 0:
                    continue
                
                # Calculate confusion matrix elements
                tn, fp, fn, tp = cms[i].ravel()
                
                # Calculate metrics
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                specificity = tn / (tn + fp) if tn + fp > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                # ICBHI score = (sensitivity + specificity) / 2
                icbhi_score = (recall + specificity) / 2
                
                # Calculate ROC AUC
                if np.unique(y_true[:, i]).size > 1:
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                else:
                    roc_auc = 0
                
                # Store metrics
                class_metrics[i] = {
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1,
                    'specificity': specificity,
                    'auc': roc_auc,
                    'icbhi_score': icbhi_score,
                    'support': int(np.sum(y_true[:, i]))
                }
            
            # Calculate macro and weighted averages
            n_classes = len(class_metrics)
            if n_classes > 0:
                # Calculate macro average
                macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
                macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
                macro_f1 = np.mean([m['f1-score'] for m in class_metrics.values()])
                macro_specificity = np.mean([m['specificity'] for m in class_metrics.values()])
                macro_auc = np.mean([m['auc'] for m in class_metrics.values()])
                macro_icbhi = np.mean([m['icbhi_score'] for m in class_metrics.values()])
                
                # Calculate weighted average
                total_support = sum([m['support'] for m in class_metrics.values()])
                weighted_precision = sum([m['precision'] * m['support'] for m in class_metrics.values()]) / total_support if total_support > 0 else 0
                weighted_recall = sum([m['recall'] * m['support'] for m in class_metrics.values()]) / total_support if total_support > 0 else 0
                weighted_f1 = sum([m['f1-score'] * m['support'] for m in class_metrics.values()]) / total_support if total_support > 0 else 0
                weighted_specificity = sum([m['specificity'] * m['support'] for m in class_metrics.values()]) / total_support if total_support > 0 else 0
                weighted_auc = sum([m['auc'] * m['support'] for m in class_metrics.values()]) / total_support if total_support > 0 else 0
                weighted_icbhi = sum([m['icbhi_score'] * m['support'] for m in class_metrics.values()]) / total_support if total_support > 0 else 0
                
                # Calculate overall accuracy
                accuracy = np.mean(np.all(y_pred == y_true, axis=1))
                
                # Store in metrics dictionary
                metrics = {
                    'accuracy': accuracy,
                    'macro_precision': macro_precision,
                    'macro_recall': macro_recall,
                    'macro_f1': macro_f1,
                    'macro_specificity': macro_specificity,
                    'macro_auc': macro_auc,
                    'macro_icbhi_score': macro_icbhi,
                    'weighted_precision': weighted_precision,
                    'weighted_recall': weighted_recall,
                    'weighted_f1': weighted_f1,
                    'weighted_specificity': weighted_specificity,
                    'weighted_auc': weighted_auc, 
                    'weighted_icbhi_score': weighted_icbhi,
                    'class_metrics': class_metrics
                }
                
                # Print nicely formatted classification report
                print(f"{'':15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
                print("-" * 55)
                
                for class_idx, m in class_metrics.items():
                    class_name = idx_to_class[class_idx]
                    print(f"{class_idx:<3} {class_name:10} {m['precision']:10.2f} {m['recall']:10.2f} {m['f1-score']:10.2f} {m['support']:10d}")
                
                print("\n")
                print(f"{'accuracy':15} {'':<10} {'':<10} {accuracy:10.2f} {y_true.shape[0]:10d}")
                print(f"{'macro avg':15} {macro_precision:10.2f} {macro_recall:10.2f} {macro_f1:10.2f} {y_true.shape[0]:10d}")
                print(f"{'weighted avg':15} {weighted_precision:10.2f} {weighted_recall:10.2f} {weighted_f1:10.2f} {y_true.shape[0]:10d}")
            
        else:
            # For single-label classification
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Calculate specificity for each class
            specificities = {}
            for class_idx in sorted(list(set(np.concatenate([y_true, y_pred])))):
                # One-vs-all approach: current class is positive, rest are negative
                y_true_binary = (y_true == class_idx).astype(int)
                y_pred_binary = (y_pred == class_idx).astype(int)
                
                # Calculate TN and FP
                tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
                fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                
                # Specificity = TN / (TN + FP)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificities[str(class_idx)] = specificity
            
            # Calculate ICBHI Score for each class
            icbhi_scores = {}
            for class_idx in sorted(list(set(np.concatenate([y_true, y_pred])))):
                class_key = str(class_idx)
                if class_key in report and 'recall' in report[class_key] and class_key in specificities:
                    icbhi_scores[class_key] = (report[class_key]['recall'] + specificities[class_key]) / 2
            
            # Calculate ROC AUC for each class
            roc_aucs = {}
            for class_idx in sorted(list(set(np.concatenate([y_true, y_pred])))):
                # Convert to one-hot for ROC calculation
                true_one_hot = (y_true == class_idx).astype(int)
                
                # Use the predicted probability for the class
                pred_prob = y_pred_probs[:, class_idx]
                
                # Skip if only one class in true values
                if np.unique(true_one_hot).size > 1:
                    fpr, tpr, _ = roc_curve(true_one_hot, pred_prob)
                    roc_aucs[str(class_idx)] = auc(fpr, tpr)
                else:
                    roc_aucs[str(class_idx)] = 0
            
            # Calculate macro-average of specificities and ICBHI scores
            macro_specificity = np.mean(list(specificities.values()))
            macro_icbhi = np.mean(list(icbhi_scores.values()))
            macro_auc = np.mean(list(roc_aucs.values()))
            
            # Calculate weighted-average of specificities and ICBHI scores
            weighted_specificity = 0
            weighted_icbhi = 0
            weighted_auc = 0
            total_support = sum([report[str(i)]['support'] for i in sorted(list(set(np.concatenate([y_true, y_pred]))))])
            
            for class_idx in sorted(list(set(np.concatenate([y_true, y_pred])))):
                class_key = str(class_idx)
                class_support = report[class_key]['support']
                weighted_specificity += specificities[class_key] * (class_support / total_support)
                weighted_icbhi += icbhi_scores[class_key] * (class_support / total_support)
                weighted_auc += roc_aucs[class_key] * (class_support / total_support)
            
            # Add these to the report dictionary
            for class_idx in sorted(list(set(np.concatenate([y_true, y_pred])))):
                class_key = str(class_idx)
                report[class_key]['specificity'] = specificities[class_key]
                report[class_key]['icbhi_score'] = icbhi_scores[class_key]
                report[class_key]['auc'] = roc_aucs[class_key]
            
            report['macro avg']['specificity'] = macro_specificity
            report['macro avg']['icbhi_score'] = macro_icbhi
            report['macro avg']['auc'] = macro_auc
            
            report['weighted avg']['specificity'] = weighted_specificity
            report['weighted avg']['icbhi_score'] = weighted_icbhi
            report['weighted avg']['auc'] = weighted_auc
            
            # Extract metrics
            metrics = {
                'accuracy': report['accuracy'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'macro_specificity': macro_specificity,
                'macro_auc': macro_auc,
                'macro_icbhi_score': macro_icbhi,
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'weighted_specificity': weighted_specificity,
                'weighted_auc': weighted_auc, 
                'weighted_icbhi_score': weighted_icbhi,
                'class_metrics': {int(k): v for k, v in report.items() if k.isdigit()}
            }
            
            # Print nicely formatted classification report
            print(f"{'':15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
            print("-" * 55)
            
            for class_idx in sorted([int(k) for k in report.keys() if k.isdigit()]):
                class_key = str(class_idx)
                class_name = idx_to_class[class_idx] if class_idx in idx_to_class else f"Class {class_idx}"
                print(f"{class_idx:<3} {class_name:10} {report[class_key]['precision']:10.2f} {report[class_key]['recall']:10.2f} {report[class_key]['f1-score']:10.2f} {report[class_key]['support']:10d}")
            
            print("\n")
            print(f"{'accuracy':15} {'':<10} {'':<10} {report['accuracy']:10.2f} {total_support:10d}")
            print(f"{'macro avg':15} {report['macro avg']['precision']:10.2f} {report['macro avg']['recall']:10.2f} {report['macro avg']['f1-score']:10.2f} {total_support:10d}")
            print(f"{'weighted avg':15} {report['weighted avg']['precision']:10.2f} {report['weighted avg']['recall']:10.2f} {report['weighted avg']['f1-score']:10.2f} {total_support:10d}")
        
        # Print extended evaluation metrics
        print("\nExtended Evaluation Metrics:")
        print("-" * 55)
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation Precision: {metrics['weighted_precision']:.4f}")
        print(f"Validation Recall/Sensitivity: {metrics['weighted_recall']:.4f}")
        print(f"Validation F1 Score: {metrics['weighted_f1']:.4f}")
        print(f"Validation AUC: {metrics['weighted_auc']:.4f}")
        print(f"Validation Specificity: {metrics['weighted_specificity']:.4f}")
        print(f"Validation ICBHI Score: {metrics['weighted_icbhi_score']:.4f}")
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        if multi_label:
            # For multi-label, plot confusion matrix for each class
            num_classes = y_true.shape[1]
            num_cols = min(3, num_classes)
            num_rows = (num_classes + num_cols - 1) // num_cols
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            if num_rows == 1 and num_cols == 1:
                axes = np.array([[axes]])
            elif num_rows == 1 or num_cols == 1:
                axes = axes.reshape(num_rows, num_cols)
            
            for i in range(num_classes):
                row = i // num_cols
                col = i % num_cols
                
                if i < y_true.shape[1]:
                    class_name = idx_to_class[i] if i in idx_to_class else f"Class {i}"
                    
                    # Plot confusion matrix
                    sns.heatmap(
                        cms[i], 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'],
                        ax=axes[row, col]
                    )
                    axes[row, col].set_title(f'Confusion Matrix - {class_name}')
                    axes[row, col].set_ylabel('True Label')
                    axes[row, col].set_xlabel('Predicted Label')
                else:
                    # Hide unused subplots
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            cm_filename = f"{save_path}/confusion_matrix_multilabel_{timestamp}.png"
            plt.savefig(cm_filename)
            print(f"Confusion matrices saved to {cm_filename}")
            
            # Plot and save ROC curves
            plt.figure(figsize=(12, 10))
            
            for i in range(y_true.shape[1]):
                class_name = idx_to_class[i] if i in idx_to_class else f"Class {i}"
                
                # Calculate ROC curve
                if np.unique(y_true[:, i]).size > 1:
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.plot(
                        fpr, 
                        tpr, 
                        lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.2f})'
                    )
            
            # Add diagonal line and labels
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            
            roc_filename = f"{save_path}/roc_curves_multilabel_{timestamp}.png"
            plt.savefig(roc_filename)
            print(f"ROC curves saved to {roc_filename}")
            
        else:
            # For single-label, plot a single confusion matrix
            plt.figure(figsize=(10, 8))
            
            # Sort unique classes to ensure consistent ordering
            classes = sorted(list(set(np.concatenate([y_true, y_pred]))))
            class_names = [idx_to_class[i] if i in idx_to_class else f"Class {i}" for i in classes]
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_filename = f"{save_path}/confusion_matrix_{timestamp}.png"
            plt.savefig(cm_filename)
            plt.close()
            print(f"Confusion matrix saved to {cm_filename}")
            
            # Plot and save ROC curves for multi-class
            plt.figure(figsize=(12, 10))
            
            # For each class, calculate ROC and AUC
            for i, class_idx in enumerate(classes):
                class_name = idx_to_class[class_idx] if class_idx in idx_to_class else f"Class {class_idx}"
                
                # Convert to one-hot for ROC calculation
                true_one_hot = (y_true == class_idx).astype(int)
                
                # Use the predicted probability for the class
                pred_prob = y_pred_probs[:, class_idx]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(true_one_hot, pred_prob)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(
                    fpr, 
                    tpr, 
                    lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.2f})'
                )
            
            # Add diagonal line and labels
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            
            roc_filename = f"{save_path}/roc_curves_{timestamp}.png"
            plt.savefig(roc_filename)
            plt.close()
            print(f"ROC curves saved to {roc_filename}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': [
                'Accuracy',
                'Precision',
                'Recall/Sensitivity',
                'F1 Score',
                'AUC',
                'Specificity',
                'ICBHI Score'
            ],
            'Value': [
                metrics['accuracy'],
                metrics['weighted_precision'],
                metrics['weighted_recall'],
                metrics['weighted_f1'],
                metrics['weighted_auc'],
                metrics['weighted_specificity'],
                metrics['weighted_icbhi_score']
            ]
        })
        
        metrics_filename = f"{save_path}/evaluation_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"Evaluation metrics saved to {metrics_filename}")
        
        # Print the final evaluation summary
        print("\nValidation Metrics:")
        print("-" * 55)
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation Precision: {metrics['weighted_precision']:.4f}")
        print(f"Validation Recall/Sensitivity: {metrics['weighted_recall']:.4f}")
        print(f"Validation F1 Score: {metrics['weighted_f1']:.4f}")
        print(f"Validation AUC: {metrics['weighted_auc']:.4f}")
        print(f"Validation Specificity: {metrics['weighted_specificity']:.4f}")
        print(f"Validation ICBHI Score: {metrics['weighted_icbhi_score']:.4f}")
        
        return metrics
    
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        traceback.print_exc()
        return None


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
