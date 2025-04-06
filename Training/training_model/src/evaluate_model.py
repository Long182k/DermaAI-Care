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


# Around line 16-20, keep the function signature
def evaluate_model(model, test_generator, multi_label=False):
    """
    Evaluate the model on the test dataset and compute detailed metrics
    
    Args:
        model: The trained Keras model
        test_generator: Generator or tf.data.Dataset for test data
        multi_label: Whether this is a multi-label classification task
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model...")
    
    # Get predictions
    try:
        y_pred_probs = model.predict(test_generator)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Trying an alternative prediction approach...")
        # Alternative prediction approach for generators
        y_pred_probs = []
        for i in range(len(test_generator)):
            batch_x, _ = test_generator[i]
            batch_pred = model.predict(batch_x)
            y_pred_probs.append(batch_pred)
        y_pred_probs = np.vstack(y_pred_probs)
    
    # Get true labels
    if hasattr(test_generator, 'classes'):
        # For ImageDataGenerator
        y_true = tf.keras.utils.to_categorical(test_generator.classes, num_classes=y_pred_probs.shape[1])
    else:
        # For YOLODetectionGenerator or tf.data.Dataset
        y_true = []
        for i in range(len(test_generator)):
            _, batch_y = test_generator[i]
            y_true.append(batch_y)
        y_true = np.vstack(y_true)
        
        # Make sure y_true matches the shape of y_pred_probs (in case of last incomplete batch)
        if y_true.shape[0] > y_pred_probs.shape[0]:
            y_true = y_true[:y_pred_probs.shape[0]]
        elif y_true.shape[0] < y_pred_probs.shape[0]:
            y_pred_probs = y_pred_probs[:y_true.shape[0]]
    
    print(f"Predictions shape: {y_pred_probs.shape}, True labels shape: {y_true.shape}")
    
    # Calculate metrics based on task type (multi-label or single-label)
    if multi_label:
        # For multi-label classification
        # Convert probabilities to binary predictions using 0.5 threshold
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        # Calculate metrics for each class
        n_classes = y_true.shape[1]
        class_metrics = {}
        
        for i in range(n_classes):
            try:
                # Class-specific metrics - use sklearn metrics where possible
                if np.sum(y_pred[:, i]) > 0 and np.sum(y_true[:, i]) > 0:
                    class_metrics[f'class_{i}_precision'] = float(sk_precision_score(y_true[:, i], y_pred[:, i]))
                    class_metrics[f'class_{i}_recall'] = float(sk_recall_score(y_true[:, i], y_pred[:, i]))
                    class_metrics[f'class_{i}_f1'] = float(sk_f1_score(y_true[:, i], y_pred[:, i]))
                else:
                    class_metrics[f'class_{i}_precision'] = 0.0
                    class_metrics[f'class_{i}_recall'] = 0.0
                    class_metrics[f'class_{i}_f1'] = 0.0
                
                # Only calculate AUC if there are both positive and negative samples
                if 0 < np.sum(y_true[:, i]) < len(y_true[:, i]):
                    class_metrics[f'class_{i}_auc'] = float(sk_roc_auc_score(y_true[:, i], y_pred_probs[:, i]))
                    average_precision = float(average_precision_score(y_true[:, i], y_pred_probs[:, i]))
                    class_metrics[f'class_{i}_avg_precision'] = average_precision
                else:
                    class_metrics[f'class_{i}_auc'] = 0.5
                    class_metrics[f'class_{i}_avg_precision'] = 0.5
            except Exception as e:
                print(f"Error calculating metrics for class {i}: {e}")
                class_metrics[f'class_{i}_precision'] = 0.0
                class_metrics[f'class_{i}_recall'] = 0.0
                class_metrics[f'class_{i}_f1'] = 0.0
                class_metrics[f'class_{i}_auc'] = 0.5
                class_metrics[f'class_{i}_avg_precision'] = 0.5
        
        # Calculate global metrics
        hamming_loss = float(np.mean(np.not_equal(y_true, y_pred)))
        subset_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
        
        # Count samples per class
        class_counts = np.sum(y_true, axis=0)
        
        # Calculate macro averaged metrics (average of per-class metrics)
        macro_precision = np.mean([class_metrics[f'class_{i}_precision'] for i in range(n_classes)])
        macro_recall = np.mean([class_metrics[f'class_{i}_recall'] for i in range(n_classes)])
        macro_f1 = np.mean([class_metrics[f'class_{i}_f1'] for i in range(n_classes)])
        
        # Print detailed evaluation report
        print("\nMulti-label Classification Report:")
        print(f"Number of test samples: {y_true.shape[0]}")
        print(f"Number of classes: {n_classes}")
        print("\nClass distribution in test set:")
        for i in range(n_classes):
            print(f"Class {i}: {class_counts[i]} samples ({class_counts[i]/y_true.shape[0]*100:.2f}%)")
        
        print("\nClass-wise metrics:")
        for i in range(n_classes):
            print(f"Class {i} - Precision: {class_metrics[f'class_{i}_precision']:.4f}, " +
                  f"Recall: {class_metrics[f'class_{i}_recall']:.4f}, " +
                  f"F1: {class_metrics[f'class_{i}_f1']:.4f}, " +
                  f"AUC: {class_metrics[f'class_{i}_auc']:.4f}, " +
                  f"AP: {class_metrics[f'class_{i}_avg_precision']:.4f}")
        
        print("\nGlobal metrics:")
        print(f"Hamming Loss: {hamming_loss:.4f}")
        print(f"Subset Accuracy: {subset_accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        # Return all metrics
        metrics = {
            'hamming_loss': float(hamming_loss),
            'subset_accuracy': float(subset_accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            **class_metrics
        }
        
    else:
        # For single-label classification
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred)
        
        # Generate classification report
        report = classification_report(y_true_classes, y_pred, output_dict=True)
        
        # Print detailed evaluation results
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred))
        
        # Calculate ROC AUC for each class
        n_classes = y_true.shape[1]
        roc_auc = {}
        for i in range(n_classes):
            try:
                # Only calculate AUC if there are both positive and negative samples for this class
                if 0 < np.sum(y_true[:, i]) < len(y_true[:, i]):
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
                    roc_auc[f'class_{i}_auc'] = float(auc(fpr, tpr))
                else:
                    roc_auc[f'class_{i}_auc'] = 0.5
            except Exception as e:
                print(f"Error calculating AUC for class {i}: {e}")
                roc_auc[f'class_{i}_auc'] = 0.5
        
        # Calculate macro and weighted averages
        metrics = {
            'accuracy': float(report['accuracy']),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']),
            'macro_f1': float(report['macro avg']['f1-score']),
            'weighted_precision': float(report['weighted avg']['precision']),
            'weighted_recall': float(report['weighted avg']['recall']),
            'weighted_f1': float(report['weighted avg']['f1-score']),
            **{f'class_{i}_precision': float(report[str(i)]['precision']) for i in range(n_classes) if str(i) in report},
            **{f'class_{i}_recall': float(report[str(i)]['recall']) for i in range(n_classes) if str(i) in report},
            **{f'class_{i}_f1': float(report[str(i)]['f1-score']) for i in range(n_classes) if str(i) in report},
            **roc_auc
        }
    
    # Save plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Generate and save confusion matrix visualization for either case
    try:
        plt.figure(figsize=(10, 8))
        if multi_label:
            # For multi-label, create a different visualization
            plt.imshow(y_true, aspect='auto', interpolation='nearest', cmap='Blues', alpha=0.5)
            plt.imshow(y_pred, aspect='auto', interpolation='nearest', cmap='Reds', alpha=0.5)
            plt.xlabel('Class')
            plt.ylabel('Sample')
            plt.title('Multi-label Classification Results\nBlue: True, Red: Predicted')
            plt.savefig('plots/multi_label_results.png', dpi=300, bbox_inches='tight')
        else:
            # For single-label, create a standard confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=range(n_classes), 
                        yticklabels=range(n_classes))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix visualization: {e}")
    
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
