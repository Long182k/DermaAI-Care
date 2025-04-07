import os
import argparse
import tensorflow as tf
import numpy as np
import random
import gc
import mlflow

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
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/1000-annotated-images/images',
                        help='Directory containing detected images')
    parser.add_argument('--labels_dir', type=str, default='/kaggle/input/1000-annotated-images/labels',
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
        
        # Fine-tune if requested
        if args.fine_tune:
            print("Fine-tuning model...")
            try:
                fine_tune_history = fine_tune_model(
                    model,
                    train_generator,
                    val_generator,
                    args.fine_tune_epochs,
                    args.early_stopping,
                    multi_label=True,
                    class_weights=class_weights,
                    batch_size=args.batch_size
                )
                
                history = fine_tune_history
                
                # Save the fine-tuned model
                try:
                    model.save(model_save_path.replace('.keras', '_fine_tuned.keras'))
                    print(f"Fine-tuned model saved")
                except Exception as save_error:
                    print(f"Error saving fine-tuned model: {save_error}")
                
                if args.log_to_mlflow:
                    try:
                        log_model_to_mlflow(model, fine_tune_history, "fine_tuned_skin_lesion_classifier", args.fold_idx, diagnosis_to_idx)
                    except Exception as mlflow_error:
                        print(f"Error logging fine-tuned model to MLflow: {mlflow_error}")
            except Exception as fine_tune_error:
                print(f"Error during fine-tuning: {fine_tune_error}")
                traceback.print_exc()
        
        # Evaluate the model
        print("Evaluating model...")
        try:
            # Use the updated evaluate_model function with proper parameters
            metrics = evaluate_model(
                model, 
                val_generator, 
                class_indices=diagnosis_to_idx,
                save_path="/kaggle/working/DermaAI-Care/Training/training_model/models"
            )
            
            # Log evaluation metrics to MLflow if requested
            if args.log_to_mlflow and metrics:
                try:
                    with mlflow.start_run(run_name=f"evaluation_fold_{args.fold_idx}"):
                        # Log comprehensive metrics
                        mlflow.log_metric("val_accuracy", metrics['accuracy'])
                        mlflow.log_metric("val_precision", metrics['weighted_precision'])
                        mlflow.log_metric("val_recall", metrics['weighted_recall'])
                        mlflow.log_metric("val_f1", metrics['weighted_f1'])
                        mlflow.log_metric("val_auc", metrics['weighted_auc'])
                        mlflow.log_metric("val_specificity", metrics['weighted_specificity'])
                        mlflow.log_metric("val_icbhi_score", metrics['weighted_icbhi_score'])
                        
                        # Log confusion matrix and ROC curve images
                        for artifact_path in [
                            f"/kaggle/working/DermaAI-Care/Training/training_model/models/confusion_matrix_*.png",
                            f"/kaggle/working/DermaAI-Care/Training/training_model/models/roc_curves_*.png"
                        ]:
                            try:
                                import glob
                                for file_path in glob.glob(artifact_path):
                                    mlflow.log_artifact(file_path)
                            except Exception as artifact_error:
                                print(f"Error logging artifact {artifact_path}: {artifact_error}")
                except Exception as mlflow_error:
                    print(f"Error logging metrics to MLflow: {mlflow_error}")
                    traceback.print_exc()
            
            # Print final summary
            if metrics:
                print("\nFinal Model Evaluation Summary:")
                print("=" * 60)
                print(f"Validation Accuracy:         {metrics['accuracy']:.4f}")
                print(f"Validation Precision:        {metrics['weighted_precision']:.4f}")
                print(f"Validation Recall/Sensitivity: {metrics['weighted_recall']:.4f}")
                print(f"Validation F1 Score:         {metrics['weighted_f1']:.4f}")
                print(f"Validation AUC:              {metrics['weighted_auc']:.4f}")
                print(f"Validation Specificity:      {metrics['weighted_specificity']:.4f}")
                print(f"Validation ICBHI Score:      {metrics['weighted_icbhi_score']:.4f}")
                print("=" * 60)
            
        except Exception as eval_error:
            print(f"Error during evaluation: {eval_error}")
            traceback.print_exc()
            
    except Exception as training_error:
        print(f"Error in main training process: {training_error}")
        traceback.print_exc()
    
    # Clean up to prevent memory leaks
    tf.keras.backend.clear_session()
    gc.collect()
    print("Training completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()