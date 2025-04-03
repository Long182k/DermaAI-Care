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
    parser.add_argument('--csv_path', type=str, default='/kaggle/input/isic-2020-training-groundtruth-v2/ISIC_2020_Training_GroundTruth_v2.csv',
                        help='Path to CSV file with image metadata')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/100-images/exp',
                        help='Directory containing detected images')
    parser.add_argument('--labels_dir', type=str, default='/kaggle/input/100-images/exp/labels',
                        help='Directory containing YOLO label files')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='Index of the fold to use for validation')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum number of samples per class')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--reduce_lr', type=int, default=3,
                        help='Patience for learning rate reduction')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine-tune the model after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--memory_limit', type=int, default=None,
                        help='GPU memory limit in MB (e.g., 4096 for 4GB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_to_mlflow', action='store_true',
                        help='Whether to log metrics to MLflow')
    
    parser.add_argument('--class_weight_multiplier', type=float, default=2.0,
                        help='Multiplier for minority class weights')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss instead of categorical crossentropy')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--augmentation_strength', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Strength of data augmentation')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Configure GPU memory settings
    strategy = setup_gpu(memory_limit=args.memory_limit)
    
    # Analyze dataset to get class weights
    print("Analyzing dataset...")
    dataset_stats = analyze_yolo_dataset(
        args.csv_path,
        args.image_dir,
        args.labels_dir
    )
    
    # Create data generators
    print(f"Creating data generators for fold {args.fold_idx}...")
    train_generator, val_generator, diagnosis_to_idx, n_classes, n_folds, train_class_indices = create_yolo_generators(
        args.csv_path,
        args.image_dir,
        args.labels_dir,
        batch_size=args.batch_size,
        min_samples_per_class=args.min_samples,
        n_folds=args.n_folds,
        fold_idx=args.fold_idx,
        seed=args.seed
    )
    
    # Build the model
    print("Building model...")
    model = build_peft_model(n_classes)
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("Training model...")
    
    # Try to create optimized datasets
    use_optimized_datasets = False
    try:
        from src.data_preprocessing import create_optimized_dataset
        print("Creating optimized datasets...")
        train_dataset = create_optimized_dataset(train_generator, is_training=True)
        val_dataset = create_optimized_dataset(val_generator, is_training=False)
        use_optimized_datasets = True
        print("Successfully created optimized datasets")
    except Exception as e:
        print(f"Could not create optimized datasets: {e}")
        print("Falling back to generator-based training")
    
    # STEP 1: TRAIN
    # In the main function, update the train_model call
    history = train_model(
        model,
        train_generator,
        val_generator,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        reduce_lr_patience=args.reduce_lr,
        class_weights=dataset_stats['class_weights'],
        train_class_indices=train_class_indices,
        class_weight_multiplier=args.class_weight_multiplier,
        use_focal_loss=args.use_focal_loss,
        learning_rate=args.learning_rate
    )
    
    # Log initial model to MLflow
    if args.log_to_mlflow:
        log_model_to_mlflow(model, history, "skin_lesion_classifier", args.fold_idx, train_class_indices)
    
    # STEP 2: FINE-TUNE (if requested)
    if args.fine_tune:
        print("Fine-tuning model...")
        fine_tune_history = fine_tune_model(
            model,
            train_generator,
            val_generator,
            epochs=args.fine_tune_epochs,
            early_stopping_patience=args.early_stopping
        )
        
        # Update history with fine-tuning results
        history = fine_tune_history
        
        # Log fine-tuned model to MLflow
        if args.log_to_mlflow:
            log_model_to_mlflow(model, fine_tune_history, "fine_tuned_skin_lesion_classifier", args.fold_idx, train_class_indices)
    
    # STEP 3: EVALUATE
    print("\n===== Model Evaluation =====")
    evaluation_metrics = evaluate_model(model, val_generator)
    
    # Log evaluation metrics to MLflow
    if args.log_to_mlflow:
        with mlflow.start_run(run_name=f"evaluation_fold_{args.fold_idx}"):
            for metric_name, metric_value in evaluation_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
    
    # STEP 4: SAVE MODEL
    # Save the model using both the fold-specific path and the standard path
    model_path = f"models/model_fold_{args.fold_idx}.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Also save to the standard path
    model.save(MODEL_SAVE_PATH)
    print(f"Model also saved to {MODEL_SAVE_PATH}")
    
    # If fine-tuned, save with a different name
    if args.fine_tune:
        fine_tuned_model_path = f"models/fine_tuned_model_fold_{args.fold_idx}.keras"
        model.save(fine_tuned_model_path)
        print(f"Fine-tuned model saved to {fine_tuned_model_path}")
    
    # Clean up
    gc.collect()
    tf.keras.backend.clear_session()
    
    print("Training, evaluation, and saving completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()