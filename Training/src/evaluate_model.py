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
    recall_score
)

def evaluate_model(model, val_generator, fold_number):
    """
    Evaluate the model and display results
    Returns a dictionary of metrics for cross-validation analysis
    """
    # Get predictions
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Convert y_true to numerical labels using diagnosis_to_idx
    y_true = np.array([val_generator.diagnosis_to_idx[diag] for diag in val_generator.diagnoses])
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix and save to file
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold_number}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'models/confusion_matrix_fold_{fold_number}.png')  # Save the plot to a file
    
    # Display the confusion matrix
    plt.show()  # Show the plot on the screen
    plt.close()  # Close the plot to free up resources
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # For AUC, we need to binarize the output (one-vs-rest approach)
    n_classes = len(np.unique(y_true))
    y_true_bin = np.eye(n_classes)[y_true]
    
    # Calculate AUC if possible (requires probabilities)
    try:
        auc = roc_auc_score(y_true_bin, predictions, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        auc = 0
    
    # Calculate sensitivity (same as recall) and specificity
    sensitivity = recall
    
    # Calculate specificity (true negative rate)
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
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'icbhi_score': icbhi_score
    }
    
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