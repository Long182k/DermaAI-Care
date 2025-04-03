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
            y_true = np.array([val_generator.diagnosis_to_idx[diag] for diag in val_generator.diagnoses])
        else:
            # Try to extract labels by iterating through the generator
            y_true = []
            for i in range(len(val_generator)):
                _, batch_y = val_generator[i]
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    # One-hot encoded
                    batch_y_classes = np.argmax(batch_y, axis=1)
                else:
                    # Already class indices
                    batch_y_classes = batch_y
                y_true.extend(batch_y_classes)
            y_true = np.array(y_true)
        
        # Check if lengths match, if not, adjust
        if len(y_true) != len(y_pred):
            print(f"Warning: Mismatch in sample counts. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
            
            # Adjust to the smaller size
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            print(f"Adjusted to {min_len} samples for evaluation")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Trying alternative approach...")
        
        # Alternative approach: manually iterate through the generator
        y_true = []
        y_pred = []
        
        # For custom generators that support indexing
        if hasattr(val_generator, '__getitem__'):
            for i in range(len(val_generator)):
                batch_x, batch_y = val_generator[i]
                batch_pred = model.predict(batch_x, verbose=0)
                
                # Convert predictions to class indices
                batch_pred_classes = np.argmax(batch_pred, axis=1)
                
                # Convert one-hot encoded y to class indices if needed
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    batch_y_classes = np.argmax(batch_y, axis=1)
                else:
                    batch_y_classes = batch_y
                
                y_true.extend(batch_y_classes)
                y_pred.extend(batch_pred_classes)
        else:
            print("Error: Generator doesn't support required operations for evaluation")
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "auc": 0
            }
    
    # Ensure we have data to evaluate
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Error: No samples available for evaluation")
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "auc": 0
        }
    
    # After getting y_true and y_pred, before printing classification report
    # Calculate and plot ROC curve with AUC for each class
    plt.figure(figsize=(12, 10))
    
    # Get the number of classes
    n_classes = len(np.unique(y_true))
    
    # For storing class-specific AUC values
    class_auc_values = {}
    
    # Calculate ROC curve and AUC for each class
    for i in range(n_classes):
        # Get class name if available
        if hasattr(val_generator, 'class_indices'):
            class_indices = {v: k for k, v in val_generator.class_indices.items()}
            class_name = class_indices.get(i, f"Class {i}")
        else:
            class_name = f"Class {i}"
        
        # Binarize the labels for one-vs-rest ROC calculation
        y_true_bin = np.array([1 if y == i else 0 for y in y_true])
        
        # Get probability scores for this class
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            y_score = predictions[:, i]
        else:
            # For binary classification
            y_score = predictions if i == 1 else 1 - predictions
        
        # Calculate ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
        
        # Calculate AUC using trapezoidal rule
        # This is the manual calculation as described in the blog
        auc_manual = 0
        for j in range(1, len(fpr)):
            # Calculate the area of the trapezoid
            width = fpr[j] - fpr[j-1]
            height = (tpr[j] + tpr[j-1]) / 2
            auc_manual += width * height
        
        # Also calculate using sklearn for verification
        auc_sklearn = roc_auc_score(y_true_bin, y_score)
        
        # Store the AUC value
        class_auc_values[class_name] = auc_sklearn
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, lw=2, 
                 label=f'{class_name} (AUC = {auc_sklearn:.3f}, Manual AUC = {auc_manual:.3f})')
    
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    # Save the ROC curve plot
    plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))
    except Exception as e:
        print(f"Error generating classification report: {e}")
        report = {}

    # Plot confusion matrix and save to file
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("models/confusion_matrix.png")  # Save the plot to a file
    plt.close()  # Close the plot to free up resources

    # Calculate metrics with zero_division=0 to avoid warnings
    try:
        # For multi-class, we need to use different averaging methods
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Also calculate metrics with macro averaging to better assess performance on minority classes
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Print additional metrics
        print(f"\nMacro-averaged Metrics (better for imbalanced classes):")
        print(f"Macro Precision: {precision_macro*100:.2f}%")
        print(f"Macro Recall: {recall_macro*100:.2f}%")
        print(f"Macro F1 Score: {f1_macro*100:.2f}%")
        
        # For AUC, we need to binarize the output (one-vs-rest approach)
        n_classes = len(np.unique(y_true))
        y_true_bin = np.eye(n_classes)[y_true]

        # Calculate AUC if possible (requires probabilities)
        try:
            auc = roc_auc_score(
                y_true_bin, predictions, multi_class="ovr", average="weighted"
            )
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            auc = 0

        # Calculate sensitivity (same as recall) and specificity
        # For multiclass, we calculate macro average
        sensitivity = recall

        # Calculate specificity (true negative rate)
        # For multiclass, we use a one-vs-rest approach
        specificities = []
        for i in range(n_classes):
            true_neg = np.sum((y_true != i) & (y_pred != i))
            actual_neg = np.sum(y_true != i)
            specificities.append(true_neg / actual_neg if actual_neg > 0 else 0)
        specificity = np.mean(specificities)

        # Calculate ICBHI score (average of sensitivity and specificity)
        icbhi_score = (sensitivity + specificity) / 2

        # Print metrics
        print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
        print(f"Validation Precision: {precision*100:.2f}%")
        print(f"Validation Recall/Sensitivity: {sensitivity*100:.2f}%")
        print(f"Validation F1 Score: {f1*100:.2f}%")
        print(f"Validation AUC: {auc*100:.2f}%")
        print(f"Validation Specificity: {specificity*100:.2f}%")
        print(f"Validation ICBHI Score: {icbhi_score*100:.2f}%")

        # Return metrics for cross-validation
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "f1": f1,
            "auc": auc,
            "specificity": specificity,
            "icbhi_score": icbhi_score,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "class_auc": class_auc_values,
            "macro_auc": np.mean(list(class_auc_values.values())) if n_classes > 2 else list(class_auc_values.values())[0]
        }
        
        if n_classes > 2:
            metrics["weighted_auc"] = roc_auc_score(y_true_bin, predictions, multi_class="ovr", average="weighted")
        
        # Find optimal threshold for balanced performance if binary classification
        if n_classes == 2:
            try:
                # Get the probabilities for the positive class
                y_prob = predictions[:, 1]
                
                # Calculate precision-recall curve
                precision_curve, recall_curve, thresholds = precision_recall_curve(y_true_bin[:, 1], y_prob)
                
                # Find threshold that maximizes F1 score
                f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                if len(thresholds) > optimal_idx:  # Ensure we have a valid threshold
                    optimal_threshold = thresholds[optimal_idx]
                    print(f"Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
                    
                    # Apply optimal threshold to predictions
                    y_pred_adjusted = (y_prob >= optimal_threshold).astype(int)
                    
                    # Recalculate metrics with adjusted threshold
                    print("\nMetrics with adjusted threshold:")
                    print(classification_report(y_true_bin[:, 1], y_pred_adjusted))
                    
                    # Add adjusted metrics to the returned dictionary
                    adjusted_accuracy = accuracy_score(y_true_bin[:, 1], y_pred_adjusted)
                    adjusted_precision = precision_score(y_true_bin[:, 1], y_pred_adjusted, zero_division=0)
                    adjusted_recall = recall_score(y_true_bin[:, 1], y_pred_adjusted, zero_division=0)
                    adjusted_f1 = f1_score(y_true_bin[:, 1], y_pred_adjusted, zero_division=0)
                    
                    metrics.update({
                        "adjusted_threshold": optimal_threshold,
                        "adjusted_accuracy": adjusted_accuracy,
                        "adjusted_precision": adjusted_precision,
                        "adjusted_recall": adjusted_recall,
                        "adjusted_f1": adjusted_f1
                    })
                    
                    print(f"Adjusted Accuracy: {adjusted_accuracy*100:.2f}%")
                    print(f"Adjusted Precision: {adjusted_precision*100:.2f}%")
                    print(f"Adjusted Recall: {adjusted_recall*100:.2f}%")
                    print(f"Adjusted F1 Score: {adjusted_f1*100:.2f}%")
            except Exception as e:
                print(f"Warning: Could not calculate optimal threshold: {e}")
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "auc": 0,
            "specificity": 0,
            "icbhi_score": 0,
            "precision_macro": 0,
            "recall_macro": 0,
            "f1_macro": 0,
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
