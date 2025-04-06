import os
import argparse
import tensorflow as tf
import numpy as np
import random
import gc
import mlflow

# Configure GPU memory growth before any other GPU operations
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

# Call this function at the very beginning
set_memory_growth()

# Import custom modules
from src.data_preprocessing import create_yolo_generators, analyze_yolo_dataset, set_random_seeds
from src.model_training import build_peft_model, build_peft_model_with_metadata, train_model, fine_tune_model, setup_gpu, log_model_to_mlflow, create_ensemble_model
from src.evaluate_model import evaluate_model, save_metrics_to_json
from sklearn.utils.class_weight import compute_class_weight

# Define model save paths
MODEL_SAVE_PATH = "/kaggle/working/DermaAI-Care/Training/training_model/models/skin_cancer_prediction_model.keras"
CHECKPOINT_PATH = "/kaggle/working/DermaAI-Care/Training/training_model/models/checkpoint.keras"  # Path to load pre-trained model

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('metrics', exist_ok=True)  # Add directory for metrics

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
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the model')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning')
    parser.add_argument('--early_stopping', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--reduce_lr', type=int, default=3, help='Patience for learning rate reduction')
    parser.add_argument('--fold_idx', type=int, default=0, help='Fold index for cross-validation')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--augmentation', type=str, default='medium', choices=['light', 'medium', 'strong'], 
                        help='Augmentation strength')
    parser.add_argument('--min_samples', type=int, default=5, help='Minimum samples per class')
    parser.add_argument('--memory_limit', type=int, default=14336, help='GPU memory limit in MB')
    parser.add_argument('--use_metadata', action='store_true', help='Whether to use metadata features')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--class_weight_multiplier', type=float, default=3.0, 
                        help='Multiplier for minority class weights')
    parser.add_argument('--use_focal_loss', action='store_true', help='Whether to use focal loss')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')

    parser.add_argument('--csv_path', type=str, 
                        default="/kaggle/input/2019-isic-csv/ISIC_2019_Training_GroundTruth.csv",
                        help='Path to the ground truth CSV file')
    parser.add_argument('--metadata_csv_path', type=str,
                        default="/kaggle/input/2019-isic-csv/ISIC_2019_Training_Metadata.csv",
                        help='Path to the metadata CSV file')
    parser.add_argument('--image_dir', type=str,
                        default="/kaggle/input/2019-isic/exp",
                        help='Directory containing the images')
    parser.add_argument('--labels_dir', type=str,
                        default="/kaggle/input/2019-isic/exp/labels",
                        help='Directory containing YOLO labels')
    parser.add_argument('--model_save_path', type=str,
                        default="/kaggle/working/DermaAI-Care/Training/training_model/models/skin_cancer_prediction_model.keras",
                        help='Path to save the final model')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Configure GPU
    strategy = setup_gpu(memory_limit=args.memory_limit)
    
    # Enable memory growth to prevent OOM errors
    set_memory_growth()
    
    # Use the path arguments instead of hardcoded paths
    # Create data generators with metadata
    train_generator, val_generator, class_indices = create_yolo_generators(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        labels_dir=args.labels_dir,
        batch_size=args.batch_size,
        min_samples_per_class=args.min_samples,
        fold_idx=args.fold_idx,
        n_folds=args.n_folds,
        seed=args.seed,
        augmentation_strength=args.augmentation,
        metadata_csv_path=args.metadata_csv_path if args.use_metadata else None,
    )
    
    # Analyze dataset to understand class distribution
    # Fix: Call analyze_yolo_dataset with the correct parameters
    analyze_yolo_dataset(args.csv_path, args.image_dir, args.labels_dir)
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.diagnoses),
        y=train_generator.diagnoses
    )
    
    # Convert to dictionary
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Build model with metadata if specified
    if args.use_metadata:
        metadata_dim = train_generator.metadata_dim
        print(f"Using metadata with {metadata_dim} features")
        model = build_peft_model_with_metadata(num_classes=len(class_indices), metadata_dim=metadata_dim)
    else:
        print(f"Using build_peft_model without metadata")
        model = build_peft_model(num_classes=len(class_indices))
    
    # Train the model - pass additional parameters
    history = train_model(
        model=model,
        train_data=train_generator,
        val_data=val_generator,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        reduce_lr_patience=args.reduce_lr,
        class_weights=class_weight_dict,
        train_class_indices=class_indices,
        class_weight_multiplier=args.class_weight_multiplier,
        use_focal_loss=args.use_focal_loss,
        learning_rate=args.learning_rate
    )
    
    # Check if training was successful
    if history is not None:
        # Evaluate the model
        metrics = evaluate_model(model, val_generator)
        
        # Save metrics to JSON file
        save_metrics_to_json(metrics, f"metrics/base_model_fold_{args.fold_idx}.json")
    else:
        print("Training failed. Saving empty metrics.")
        save_metrics_to_json({"error": "Training failed"}, f"metrics/base_model_fold_{args.fold_idx}.json")
    
    # Save the model
    try:
        model_path = f"models/model_fold_{args.fold_idx}.keras"
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Log model to MLflow with metadata information
    log_model_to_mlflow(
        model, 
        history, 
        "skin_lesion_classifier", 
        args.fold_idx,
        class_indices,
        metadata_used=args.use_metadata
    )
    
    # Fine-tune if requested
    if args.fine_tune:
        print("\nStarting fine-tuning...")
        fine_tuned_model = fine_tune_model(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=args.fine_tune_epochs,
            early_stopping_patience=args.early_stopping,
            learning_rate=args.learning_rate / 10 
        )
        
        # Evaluate fine-tuned model
        fine_tuned_metrics = evaluate_model(fine_tuned_model, val_generator)
        
        # Save fine-tuned metrics
        save_metrics_to_json(fine_tuned_metrics, f"metrics/fine_tuned_model_fold_{args.fold_idx}.json")
        
        # Save fine-tuned model
        fine_tuned_model_path = f"models/fine_tuned_model_fold_{args.fold_idx}.keras"
        fine_tuned_model.save(fine_tuned_model_path)
        
        # Create ensemble model from base and fine-tuned models
        print("\nCreating ensemble model...")
        model_paths = [model_path, fine_tuned_model_path]
        
        # If previous fold models exist, add them to the ensemble
        for prev_fold in range(args.fold_idx):
            prev_model_path = f"models/fine_tuned_model_fold_{prev_fold}.keras"
            if os.path.exists(prev_model_path):
                model_paths.append(prev_model_path)
                print(f"Adding previous fold model: {prev_model_path}")
        
        # Create ensemble with available models (up to 3)
        ensemble_model = create_ensemble_model(
            model_paths=model_paths[:3],  # Use up to 3 models
            num_classes=len(class_indices)
        )
        
        # Evaluate ensemble model
        ensemble_metrics = evaluate_model(ensemble_model, val_generator)
        
        # Save ensemble metrics
        save_metrics_to_json(ensemble_metrics, f"metrics/ensemble_model_fold_{args.fold_idx}.json")
        
        # Save the final model for deployment
        ensemble_model.save(args.model_save_path)
        
        print(f"Training, evaluation, and saving completed successfully!")

if __name__ == "__main__":
    main()