from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def evaluate_model(model, val_generator):
    """
    Evaluate the model and display results
    Returns a dictionary of metrics for cross-validation analysis
    """
    # Get predictions
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Calculate and print metrics
    loss, accuracy, auc, precision, recall = model.evaluate(val_generator)
    print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")
    
    # Return metrics for cross-validation
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': report['weighted avg']['f1-score']
    }
    
    return metrics

def save_model(model, filepath):
    """
    Save the trained model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        model.save(filepath)
        print(f"\nModel successfully saved to: {filepath}")
        
        # Save model summary
        summary_path = os.path.join(os.path.dirname(filepath), 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise 