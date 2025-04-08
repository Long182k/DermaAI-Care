import os
import argparse
import tensorflow as tf
import numpy as np
import random
import gc
import mlflow
import traceback

# Import custom modules
from src.data_preprocessing import create_yolo_generators, analyze_yolo_dataset, set_random_seeds
from src.model_training import build_peft_model, train_model, fine_tune_model, setup_gpu, log_model_to_mlflow
from src.evaluate_model import evaluate_model  # Make sure this import is correct

# Define model save paths
MODEL_SAVE_PATH = "/kaggle/working/DermaAI-Care/Training/training_model/models/skin_cancer_prediction_model.keras"
CHECKPOINT_PATH = "/kaggle/working/DermaAI-Care/Training/training_model/models/checkpoint.keras"  # Path to load pre-trained model

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def set_memory_growth():
    """Configure GPU to allow memory growth to prevent OOM errors"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled on {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train skin lesion classification model')
    parser.add_argument('--csv_path', type=str, default='/kaggle/input/2019-isic-csv/ISIC_2019_Training_GroundTruth.csv',
                        help='Path to CSV file with image metadata')
    parser.add_argument('--metadata_csv_path', type=str, default="/kaggle/input/2019-isic-csv/ISIC_2019_Training_Metadata.csv",
                        help='Path to CSV file with patient metadata (age, sex, anatomical site)')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/2019-isic/exp',
                        help='Directory containing detected images')
    parser.add_argument('--labels_dir', type=str, default='/kaggle/input/2019-isic/exp/labels',
                        help='Directory containing YOLO label files')
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
    parser.add_argument('--memory_limit', type=int, default=None,
                        help='GPU memory limit in MB (e.g., 4096 for 4GB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_to_mlflow', action='store_true',
                        help='Whether to log metrics to MLflow')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--augmentation_strength', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Strength of data augmentation')
    parser.add_argument('--use_metadata', action='store_true',
                        help='Whether to use patient metadata (age, sex, anatomical site)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    try:
        # Set random seeds for reproducibility
        set_random_seeds(args.seed)
        
        # Configure GPU memory settings
        print("Setting up GPU...")
        tf_strategy = setup_gpu(memory_limit=args.memory_limit)
        print(f"Using strategy: {tf_strategy}")
        
        # Make sure we clear any existing session before proceeding
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Update model save path if provided
        model_save_path = MODEL_SAVE_PATH
        if args.model_save_path:
            model_save_path = args.model_save_path
            print(f"Custom model save path set: {model_save_path}")
        
        # Analyze dataset to get class weights and statistics
        print("Analyzing dataset...")
        dataset_stats = analyze_yolo_dataset(
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
            metadata_csv_path=args.metadata_csv_path if args.use_metadata else None
        )
        
        # Build the model with multi-label output
        print("Building model...")
        model = build_peft_model(n_classes, multi_label=True)
        
        if model is None:
            print("Failed to build model. Exiting.")
            return
        
        # Print model summary
        model.summary()
        
        # Train the model
        print("Training model...")
        try:
            history = train_model(
                model,
                train_generator,
                val_generator,
                args.epochs,
                args.early_stopping,
                args.reduce_lr,
                class_weights,
                args.learning_rate,
                multi_label=True,
                batch_size=args.batch_size
            )
            
            # Save the model
            try:
                model.save(model_save_path)
                print(f"Model saved to {model_save_path}")
            except Exception as save_error:
                print(f"Error saving model: {save_error}")
                traceback.print_exc()
            
            # Log initial model to MLflow
            if args.log_to_mlflow:
                try:
                    log_model_to_mlflow(model, history, "skin_lesion_classifier", args.fold_idx, diagnosis_to_idx)
                except Exception as mlflow_error:
                    print(f"Error logging to MLflow: {mlflow_error}")
            
            # Fine-tune the model if requested
            if args.fine_tune:
                print("\nFine-tuning model...")
                fine_tune_history, fine_tuned_model = fine_tune_model(
                    model,
                    train_generator,
                    val_generator,
                    epochs=args.epochs,
                    early_stopping_patience=args.early_stopping,
                    multi_label=True,
                    class_weights=class_weights,
                    batch_size=args.batch_size,
                    model_save_path=model_save_path
                )
                
                if fine_tuned_model is not None:
                    model = fine_tuned_model  # Use the fine-tuned model
                    print("Fine-tuned model saved")
                else:
                    print("Fine-tuning failed, using original model")
                
                # Log fine-tuning metrics to MLflow
                if fine_tune_history is not None:
                    log_model_to_mlflow(model, fine_tune_history, "fine_tuned_model", 0, diagnosis_to_idx)
            
            # Evaluate model
            print("\nEvaluating model...")
            # Ensure evaluation directory exists
            eval_dir = os.path.join(os.path.dirname(model_save_path), "evaluation_results")
            os.makedirs(eval_dir, exist_ok=True)
            
            metrics = evaluate_model(
                model=model,
                test_generator=val_generator,
                class_names=list(diagnosis_to_idx.keys()),
                output_dir=eval_dir
            )
            
            if metrics:
                print("\nEvaluation Metrics:")
                print("-" * 55)
                print(f"Validation Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"Validation Precision: {metrics.get('precision', 0):.4f}")
                print(f"Validation Recall/Sensitivity: {metrics.get('recall', 0):.4f}")
                print(f"Validation F1 Score: {metrics.get('f1', 0):.4f}")
                print(f"Validation AUC: {metrics.get('auc', 0):.4f}")
                print(f"Validation Specificity: {metrics.get('specificity', 0):.4f}")
                print(f"Validation ICBHI Score: {metrics.get('icbhi_score', 0):.4f}")
                
                # Print paths of saved files
                if 'confusion_matrix_path' in metrics:
                    print(f"\nConfusion matrix saved to: {metrics['confusion_matrix_path']}")
                if 'report_path' in metrics:
                    print(f"Classification report saved to: {metrics['report_path']}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error in main training process: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()