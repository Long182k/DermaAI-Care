import os
import argparse
import tensorflow as tf
import numpy as np
import gc
import traceback
import time
from datetime import datetime

# Import custom modules
from src.data_preprocessing import create_yolo_generators, analyze_yolo_dataset, set_random_seeds
from src.model_training import setup_gpu, log_model_to_mlflow, TimeoutCallback
from src.model_training_fixed import train_model, fine_tune_model, build_peft_model  # Import the fixed functions
from src.evaluate_model import evaluate_model
from src.model_utils import calculate_adaptive_thresholds, apply_adaptive_thresholds, correct_bias, detect_prediction_bias, calculate_class_weights

# Define model save paths
MODEL_SAVE_PATH = "/kaggle/working/DermaAI-Care/Training/training_model/models/skin_cancer_prediction_model.keras"

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Define custom thresholds for different classes to handle imbalance
def get_adaptive_thresholds(class_weights, base_threshold=0.5):
    """
    Generate adaptive thresholds based on class weights
    
    Args:
        class_weights: Dictionary mapping class indices to weights
        base_threshold: Base threshold to adjust from
    
    Returns:
        Dictionary mapping class indices to custom threshold values
    """
    if not class_weights:
        return {}
    
    # Get min and max weights for normalization
    min_weight = min(class_weights.values())
    max_weight = max(class_weights.values())
    weight_range = max_weight - min_weight
    
    # Set thresholds - higher weight (rarer class) gets lower threshold
    thresholds = {}
    for class_idx, weight in class_weights.items():
        if weight_range > 0:
            # Normalize weight to 0-1 range and invert (higher weight = lower threshold)
            norm_weight = (weight - min_weight) / weight_range
            
            # More aggressive threshold adjustment for very imbalanced datasets
            if max_weight / min_weight > 5:  # Significantly imbalanced
                # Adjust threshold down more for rare classes
                threshold = base_threshold - (norm_weight * 0.35)
            else:
                # Smaller adjustment for more balanced datasets
                threshold = base_threshold - (norm_weight * 0.2)
            
            # Ensure threshold is in reasonable range
            threshold = max(0.15, min(0.8, threshold))
            thresholds[class_idx] = threshold
        else:
            thresholds[class_idx] = base_threshold
    
    return thresholds

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train skin lesion classification model')
    parser.add_argument('--csv_path', type=str, default='/kaggle/input/isic-2019-labeled/ISIC_2019_Labeled_GroundTruth.csv',
                        help='Path to CSV file with image metadata')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/isic-2019-labeled/isic-2019-labeled/isic-2019-labeled/images',
                        help='Directory containing detected images')
    parser.add_argument('--labels_dir', type=str, default='/kaggle/input/isic-2019-labeled/isic-2019-labeled/isic-2019-labeled/labels',
                        help='Directory containing YOLO label files')
    parser.add_argument('--metadata_csv_path', type=str, default="/kaggle/input/isic-2019-labeled/ISIC_2019_Labeled_Metadata.csv",
                       help='Path to CSV file with patient metadata (age, sex, anatomical site)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='Index of the fold to use for validation')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum number of samples per class')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--reduce_lr', type=int, default=5,
                        help='Patience for learning rate reduction')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine-tune the model after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--augmentation_strength', type=str, default='high',
                        choices=['low', 'medium', 'high'],
                        help='Strength of data augmentation')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--multi_label', action='store_true', default=True,
                        help='Use multi-label classification')
    parser.add_argument('--use_metadata', action='store_true',
                       help='Whether to use patient metadata (age, sex, anatomical site)')
    parser.add_argument('--focal_loss', action='store_true', default=True,
                       help='Use focal loss for handling class imbalance')
    parser.add_argument('--fine_tune_lr', type=float, default=5e-6,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--initial_bias', type=float, default=-0.5,
                       help='Initial bias value for output layer (-0.5 recommended)')
    parser.add_argument('--weighted_metrics', action='store_true', default=True,
                       help='Use class-weighted metrics for evaluation')
    parser.add_argument('--balance_thresholds', action='store_true', default=True,
                       help='Use adaptive thresholds based on class balance')
    parser.add_argument('--extra_regularization', action='store_true', default=False,
                       help='Add extra L2 regularization to prevent overfitting')
    
    args = parser.parse_args()
    
    try:
        # Set random seeds for reproducibility
        set_random_seeds(args.seed)
        
        # Clear any existing session to avoid conflicts
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Update model save path if provided
        model_save_path = MODEL_SAVE_PATH
        if args.model_save_path:
            model_save_path = args.model_save_path
            print(f"Custom model save path set: {model_save_path}")
        
        # Create model save directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Analyze dataset
        print("Analyzing dataset...")
        analyze_yolo_dataset(
            args.csv_path,
            args.image_dir,
            args.labels_dir,
            metadata_csv_path=args.metadata_csv_path if args.use_metadata else None
        )
    
        # Create data generators
        print(f"Creating data generators for fold {args.fold_idx}...")
        train_generator, val_generator, diagnosis_to_idx, n_classes, n_folds, class_weights = create_yolo_generators(
            args.csv_path,
            args.image_dir,
            args.labels_dir,
            batch_size=args.batch_size,
            min_samples_per_class=args.min_samples,
            n_folds=args.n_folds,
            fold_idx=args.fold_idx,
            seed=args.seed,
            augmentation_strength=args.augmentation_strength,
            metadata_csv_path=args.metadata_csv_path if args.use_metadata else None,
            balance_classes=True  # Always balance classes
        )
        
        # Build the model
        print("Building model...")
        print(f"Number of classes: {n_classes}")
        model = build_peft_model(
            n_classes, 
            multi_label=args.multi_label, 
            use_focal_loss=args.focal_loss,
            initial_bias=args.initial_bias,  # Add initial bias parameter
            extra_regularization=args.extra_regularization  # Add regularization parameter
        )
        
        if model is None:
            print("Failed to build model. Exiting.")
            return
            
        # Apply bias correction to account for class imbalance
        if class_weights:
            print("Applying bias correction to account for class imbalance...")
            model = correct_bias(model, class_weights, layer_name='dense_2')  # Use the last dense layer name
        
        # Print model summary
        model.summary()
        
        # Print class distribution and weights
        print("\nClass distribution and weights:")
        print("Class Index | Class Name | Weight")
        print("-" * 40)
        for class_name, idx in diagnosis_to_idx.items():
            weight = class_weights.get(idx, 'N/A') if class_weights else 'N/A'
            if isinstance(weight, str):
                print(f"{idx:^10} | {class_name:^10} | {weight:^6}")
            else:
                print(f"{idx:^10} | {class_name:^10} | {weight:^6.3f}")
        
        # Train the model
        print("Training model...")
        try:
            # Configure timeout callback
            timeout_seconds = 14400  # 4 hours
            callbacks = [
                TimeoutCallback(timeout_seconds=timeout_seconds)
            ]
            
            # Use the fixed train_model function without workers/use_multiprocessing
            history = train_model(
                model=model,
                train_generator=train_generator,
                val_generator=val_generator,
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping,
                reduce_lr_patience=args.reduce_lr,
                callbacks=callbacks,
                class_weights=class_weights,
                learning_rate=args.learning_rate,
                multi_label=args.multi_label,
                batch_size=args.batch_size,
                model_save_path=model_save_path
            )
            
            # Save the model
            try:
                model.save(model_save_path)
                print(f"Model saved to {model_save_path}")
            except Exception as save_error:
                print(f"Error saving model: {save_error}")
                traceback.print_exc()
        
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
        
        # Fine-tune the model if requested
        if args.fine_tune:
            print("Fine-tuning model...")
            try:
                # First evaluate model to see baseline performance
                print("\nBaseline model performance before fine-tuning:")
                baseline_metrics = model.evaluate(val_generator, verbose=1)
                print(f"Baseline metrics: {dict(zip(model.metrics_names, baseline_metrics))}")
                
                # Calculate prediction distribution before fine-tuning
                print("\nAnalyzing prediction distribution before fine-tuning:")
                
                # Get a batch from the validation generator, handling both tf.data.Dataset and custom generators
                if isinstance(val_generator, tf.data.Dataset):
                    # For tf.data.Dataset, take the first batch
                    for test_batch_x, test_batch_y in val_generator.take(1):
                        break
                else:
                    # For custom generators (like keras.utils.Sequence), use indexing
                    test_batch_x, test_batch_y = val_generator[0]
                
                preds = model.predict(test_batch_x,test_batch_y)
                preds_classes = np.argmax(preds, axis=1) if preds.shape[1] > 1 else (preds > 0.5).astype(int)
                print("Prediction class distribution (sample):")
                for cls in range(min(preds.shape[1], 9)):
                    count = np.sum(preds_classes == cls)
                    print(f"  Class {cls}: {count} predictions")
                
                # Adjust fine-tuning parameters based on dataset analysis
                adaptive_fine_tune_epochs = min(30, args.fine_tune_epochs)  # Cap at 30 epochs max
                adaptive_fine_tune_lr = args.fine_tune_lr
                if np.mean(baseline_metrics) < 0.3:  # If model is performing poorly
                    adaptive_fine_tune_lr = 1e-5  # Use higher learning rate
                    print(f"Adjusting fine-tune learning rate to {adaptive_fine_tune_lr} due to low performance")
                
                # Use the fixed fine_tune_model function with model_save_path
                fine_tuned_model = fine_tune_model(
                    model=model,
                    train_generator=train_generator,
                    val_generator=val_generator,
                    diagnosis_to_idx=diagnosis_to_idx,
                    epochs=adaptive_fine_tune_epochs,
                    fine_tune_lr=adaptive_fine_tune_lr,
                    class_weights=class_weights,
                    batch_size=min(16, args.batch_size),
                    early_stopping=args.early_stopping,
                    reduce_lr=args.reduce_lr
                )
                
                # Store fine_tune_history as None since the function doesn't return it
                fine_tune_history = None
                
                if fine_tuned_model is not None:
                    # Evaluate to compare with baseline
                    print("\nFine-tuned model performance:")
                    fine_tuned_metrics = fine_tuned_model.evaluate(val_generator, verbose=1)
                    print(f"Fine-tuned metrics: {dict(zip(fine_tuned_model.metrics_names, fine_tuned_metrics))}")
                    
                    # Calculate prediction distribution after fine-tuning
                    print("\nAnalyzing prediction distribution after fine-tuning:")
                    
                    # Reuse the same test batch data that we already extracted
                    preds = fine_tuned_model.predict(test_batch_x)
                    preds_classes = np.argmax(preds, axis=1) if preds.shape[1] > 1 else (preds > 0.5).astype(int)
                    print("Prediction class distribution (sample):")
                    for cls in range(min(preds.shape[1], 9)):
                        count = np.sum(preds_classes == cls)
                        print(f"  Class {cls}: {count} predictions")
                    
                    # Only use fine-tuned model if it improved over baseline
                    if np.mean(fine_tuned_metrics) > np.mean(baseline_metrics):
                        model = fine_tuned_model  # Use the fine-tuned model
                        print("Successfully fine-tuned the model with improved metrics")
                    else:
                        print("Fine-tuning did not improve metrics, using original model")
                else:
                    print("Fine-tuning failed, using the original model for evaluation")
            except Exception as fine_tune_error:
                print(f"Error during fine-tuning: {fine_tune_error}")
                traceback.print_exc()
        
        # Generate custom prediction thresholds using model_utils function
        prediction_thresholds = {}
        if args.balance_thresholds:
            print("Generating predictions for adaptive threshold calculation...")
            # Get validation data predictions for threshold calculation
            val_data_sample = []
            val_labels_sample = []
            
            # Get a representative sample from validation data
            sample_size = min(500, len(val_generator) * val_generator.batch_size)
            samples_collected = 0
            
            for i in range(min(20, len(val_generator))):
                if samples_collected >= sample_size:
                    break
                try:
                    batch_x, batch_y = val_generator[i]
                    val_data_sample.append(batch_x)
                    val_labels_sample.append(batch_y)
                    samples_collected += len(batch_x)
                except Exception as e:
                    print(f"Error collecting validation sample: {e}")
                    continue
            
            if val_data_sample:
                val_data_sample = np.vstack(val_data_sample)
                val_labels_sample = np.vstack(val_labels_sample)
                
                # Get predictions
                val_preds = model.predict(val_data_sample, verbose=1)
                
                # Calculate adaptive thresholds using model_utils function
                print("Calculating adaptive thresholds based on validation data...")
                thresholds = calculate_adaptive_thresholds(
                    val_labels_sample, val_preds, 
                    class_weights=class_weights, 
                    method='pr'  # Use precision-recall curve for imbalanced data
                )
                
                # Convert to dictionary
                prediction_thresholds = {i: float(thresholds[i]) for i in range(len(thresholds))}
                
                # Ensure very rare classes have lower thresholds
                if class_weights:
                    for class_idx, weight in class_weights.items():
                        # Identify very rare classes
                        if weight > 2 * np.mean(list(class_weights.values())):
                            # Use an even lower threshold
                            prediction_thresholds[class_idx] = max(0.15, prediction_thresholds.get(class_idx, 0.35) - 0.1)
            else:
                # Fallback to the original method if we couldn't get validation samples
                prediction_thresholds = get_adaptive_thresholds(class_weights, base_threshold=0.35)
        else:
            # Use default threshold for all classes if not balancing
            prediction_thresholds = {i: 0.5 for i in range(n_classes)}
        
        print(f"Using custom prediction thresholds: {prediction_thresholds}")
        
        # Evaluate model
        print("Generating predictions...")
        # Create a new evaluation directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(os.path.dirname(model_save_path), 'evaluation_results', timestamp)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Get class names for evaluation
        eval_class_names = []
        for i in range(n_classes):
            # Find the class name for each index
            for diagnosis, idx in diagnosis_to_idx.items():
                if idx == i:
                    eval_class_names.append(diagnosis)
                    break
            else:
                # If we didn't find a match, use a generic name
                eval_class_names.append(f"Class_{i}")
        
        # Perform evaluation with the custom thresholds
        metrics = evaluate_model(
            model=model,
            test_generator=val_generator,
            class_names=eval_class_names,
            output_dir=eval_dir,
            prediction_thresholds=prediction_thresholds,
            use_weighted_metrics=args.weighted_metrics  # Add weighted metrics parameter
        )
        
        # Print metrics
        if metrics:
            print("\nEvaluation Metrics:")
            print("-" * 55)
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            print(f"Validation Precision: {metrics['precision']:.4f}")
            print(f"Validation Recall/Sensitivity: {metrics['recall']:.4f}")
            print(f"Validation F1 Score: {metrics['f1']:.4f}")
            print(f"Validation AUC: {metrics['auc']:.4f}")
            print(f"Validation Specificity: {metrics['specificity']:.4f}")
            
            # Print weighted metrics if they exist
            if 'weighted_accuracy' in metrics:
                print("\nWeighted Metrics (Balanced for Class Imbalance):")
                print("-" * 55)
                print(f"Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
                print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
                print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
                print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
                print(f"Weighted AUC: {metrics['weighted_auc']:.4f}")
            
            # Save confusion matrix and classification report
            print(f"\nConfusion matrix saved to: {metrics['confusion_matrix_path']}")
            print(f"Classification report saved to: {metrics['report_path']}")
            
            # If training multiple folds, save metrics for aggregation
            if args.n_folds > 1:
                # Save metrics for this fold
                fold_metrics_path = os.path.join(
                    os.path.dirname(model_save_path), 
                    f'fold_{args.fold_idx}_metrics.json'
                )
                with open(fold_metrics_path, 'w') as f:
                    import json
                    # Convert numpy values to float for JSON serialization
                    metrics_json = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                  for k, v in metrics.items() if k not in ['confusion_matrix_path', 'report_path']}
                    json.dump(metrics_json, f, indent=2)
                
                print(f"Fold {args.fold_idx} metrics saved to: {fold_metrics_path}")
        else:
            print("Evaluation failed to return metrics.")
    except Exception as e:
        print(f"Error in main training process: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()