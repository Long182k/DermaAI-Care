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
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png')  # Save the plot to a file
    plt.close()  # Close the plot to free up resources
    
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
    Save the trained model with custom objects
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model with custom objects
        model.save(filepath, save_format='h5')
        print(f"\nModel successfully saved to: {filepath}")
        
        # Save model summary
        summary_path = os.path.join(os.path.dirname(filepath), 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to: {summary_path}")
        
        # Save model architecture visualization
        try:
            from tensorflow.keras.utils import plot_model
            arch_path = os.path.join(os.path.dirname(filepath), 'model_architecture.png')
            plot_model(model, to_file=arch_path, show_shapes=True, show_layer_names=True)
            print(f"Model architecture visualization saved to: {arch_path}")
        except Exception as viz_error:
            print(f"Warning: Could not save model visualization: {viz_error}")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise 