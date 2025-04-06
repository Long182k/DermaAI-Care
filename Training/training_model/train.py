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
from src.evaluate_model import evaluate_model  # Add this import at the top

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
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the model')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning')
    parser.add_argument('--early_stopping', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--reduce_lr', type=int, default=3, help='Patience for learning rate reduction')
    parser.add_argument('--fold', type=int, default=0, help='Fold index for cross-validation')
    parser.add_argument('--augmentation', type=str, default='medium', choices=['light', 'medium', 'strong'], 
                        help='Augmentation strength')
    parser.add_argument('--min_samples', type=int, default=5, help='Minimum samples per class')
    parser.add_argument('--memory_limit', type=int, default=None, help='GPU memory limit in MB')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Configure GPU
    strategy = setup_gpu(memory_limit=args.memory_limit)
    
    # Enable memory growth to prevent OOM errors
    set_memory_growth()
    
    # Define paths
    csv_path = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/data/ISIC_2019_Training_GroundTruth.csv"
    image_dir = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/data/exp"
    labels_dir = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/data/exp/labels"
    
    # Create data generators
    train_generator, val_generator, class_indices = create_yolo_generators(
        csv_path=csv_path,
        image_dir=image_dir,
        labels_dir=labels_dir,
        batch_size=args.batch_size,
        min_samples_per_class=args.min_samples,
        fold_idx=args.fold,
        augmentation_strength=args.augmentation
    )
    
    # Analyze dataset to understand class distribution
    analyze_yolo_dataset(train_generator, val_generator)
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.diagnoses),
        y=train_generator.diagnoses
    )
    
    # Convert to dictionary
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Build model with 9 output classes
    model = build_peft_model(num_classes=9)
    
    # Train the model
    history = train_model(
        model=model,
        train_data=train_generator,
        val_data=val_generator,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        reduce_lr_patience=args.reduce_lr,
        class_weights=class_weight_dict,
        train_class_indices=class_indices,
        class_weight_multiplier=3.0,  # Increase weight for minority classes
        use_focal_loss=True
    )
    
    # Evaluate the model
    metrics = evaluate_model(model, val_generator)
    
    # Save the model
    model.save(f"models/model_fold_{args.fold}.keras")
    
    # Log model to MLflow
    log_model_to_mlflow(model, history, "skin_lesion_classifier", args.fold, class_indices)
    
    # Fine-tune if requested
    if args.fine_tune:
        print("\nStarting fine-tuning...")
        fine_tuned_model = fine_tune_model(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=args.fine_tune_epochs,
            early_stopping_patience=args.early_stopping
        )
        
        # Evaluate fine-tuned model
        fine_tuned_metrics = evaluate_model(fine_tuned_model, val_generator)
        
        # Save fine-tuned model
        fine_tuned_model.save(f"models/fine_tuned_model_fold_{args.fold}.keras")
        
        # Create ensemble model for better performance
        print("\nCreating ensemble model...")
        ensemble_model = create_ensemble_model(num_classes=9, num_models=3)
        
        # Save the final model for deployment
        ensemble_model.save(MODEL_SAVE_PATH)
        
        print(f"Training, evaluation, and saving completed successfully!")

if __name__ == "__main__":
    main()