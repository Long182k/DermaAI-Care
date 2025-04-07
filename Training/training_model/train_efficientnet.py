import os
import argparse
import tensorflow as tf
import numpy as np
import random
import gc
import mlflow
import time
from datetime import datetime

# Import custom modules
from src.efficientnet_preprocessing import create_efficientnet_generators, analyze_dataset, set_random_seeds
from src.efficientnet_model import build_efficientnet_model, fine_tune_efficientnet, create_efficientnet_ensemble, predict_with_ensemble
from src.evaluate_model import evaluate_model

# Define model save paths
MODEL_SAVE_PATH = "models/skin_cancer_efficientnet_model.keras"
CHECKPOINT_PATH = "models/efficientnet_checkpoint.keras"

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

def setup_gpu(memory_limit=None):
    """Setup GPU for training with optional memory limit"""
    # Set memory growth to avoid OOM errors
    set_memory_growth()
    
    # Create a strategy for distributed training if multiple GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs found. Using CPU strategy.")
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")
    
    # If memory limit is specified, limit GPU memory
    if memory_limit is not None and gpus:
        try:
            # Convert MB to bytes
            memory_limit_bytes = memory_limit * 1024 * 1024
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_bytes)]
            )
            print(f"GPU memory limited to {memory_limit} MB")
        except Exception as e:
            print(f"Error setting memory limit: {e}")
    
    # Use MirroredStrategy if multiple GPUs are available
    if len(gpus) > 1:
        print(f"Using MirroredStrategy with {len(gpus)} GPUs")
        return tf.distribute.MirroredStrategy()
    else:
        print("Using default strategy with 1 GPU")
        return tf.distribute.get_strategy()

# Custom callbacks for memory management and timeout protection
class TimeoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, timeout_seconds=3600):  # Default timeout of 1 hour
        super(TimeoutCallback, self).__init__()
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.timeout_seconds:
            print(f"\nStopping training due to timeout ({self.timeout_seconds} seconds)")
            self.model.stop_training = True

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def __init__(self, cleanup_frequency=2):
        super(MemoryCleanupCallback, self).__init__()
        self.cleanup_frequency = cleanup_frequency
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.cleanup_frequency == 0:
            # Perform memory cleanup
            gc.collect()
            print("\nPerformed memory cleanup")

def train_model(model, train_data, val_data, epochs, early_stopping_patience, reduce_lr_patience, 
                class_weights=None, learning_rate=0.0001, multi_label=True, batch_size=32):
    """Train the model with proper callbacks and monitoring"""
    print(f"Using batch size: {batch_size}")
    
    # Set up callbacks
    monitor_metric = 'val_loss' if multi_label else 'val_accuracy'
    mode = 'min' if monitor_metric == 'val_loss' else 'max'
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode=mode,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor=monitor_metric,
            save_best_only=True,
            mode=mode,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/efficientnet_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        ),
        TimeoutCallback(timeout_seconds=7200),  # 2 hour timeout
        MemoryCleanupCallback(cleanup_frequency=2)
    ]
    
    # Train the model
    try:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights if not multi_label else None,  # Only use for single-label
            verbose=1
        )
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def log_model_to_mlflow(model, history, model_name, fold_idx, class_indices):
    """Log model and metrics to MLflow"""
    try:
        # Set experiment
        mlflow.set_experiment("skin_lesion_classification")
        
        # Start a new run
        with mlflow.start_run(run_name=f"{model_name}_fold_{fold_idx}"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("fold_idx", fold_idx)
            mlflow.log_param("num_classes", len(class_indices))
            
            # Log metrics
            for metric in ['loss', 'accuracy', 'auc', 'precision', 'recall']:
                if metric in history.history:
                    mlflow.log_metric(f"final_{metric}", history.history[metric][-1])
                if f"val_{metric}" in history.history:
                    mlflow.log_metric(f"final_val_{metric}", history.history[f"val_{metric}"][-1])
            
            # Log model
            signature = infer_signature(np.zeros((1, 224, 224, 3)), np.zeros((1, len(class_indices))))
            mlflow.keras.log_model(model, "model", signature=signature)
            
            # Log class indices
            mlflow.log_dict(class_indices, "class_indices.json")
            
            print(f"Model logged to MLflow with run_id: {mlflow.active_run().info.run_id}")
    except Exception as e:
        print(f"Error logging to MLflow: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train skin lesion classification model with EfficientNet')
    parser.add_argument('--csv_path', type=str, default='/kaggle/input/2019-isic-csv/ISIC_2019_Training_GroundTruth.csv',
                        help='Path to CSV file with image metadata')
    parser.add_argument('--metadata_csv_path', type=str, default="/kaggle/input/2019-isic-csv/ISIC_2019_Training_Metadata.csv",
                        help='Path to CSV file with patient metadata (age, sex, anatomical site)')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/1000-annotated-images/images',
                        help='Directory containing detected images')
    parser.add_argument('--labels_dir', type=str, default='/kaggle/input/1000-annotated-images/labels',
                        help='Directory containing YOLO label files')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='Index of the fold to use for validation')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum number of samples per class')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--reduce_lr', type=int, default=8,
                        help='Patience for learning rate reduction')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine-tune the model after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=30,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--memory_limit', type=int, default=None,
                        help='GPU memory limit in MB (e.g., 4096 for 4GB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_to_mlflow', action='store_true',
                        help='Whether to log metrics to MLflow')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--augmentation_strength', type=str, default='high',
                        choices=['low', 'medium', 'high'],
                        help='Strength of data augmentation')
    parser.add_argument('--use_metadata', action='store_true',
                        help='Whether to use patient metadata (age, sex, anatomical site)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--model_size', type=str, default='B3',
                        choices=['B0', 'B3', 'B5', 'ensemble'],
                        help='EfficientNet model size or ensemble')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for input (square)')
    
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
    dataset_stats = analyze_dataset(
        args.csv_path,
        args.image_dir,
        args.labels_dir,
        metadata_csv_path=args.metadata_csv_path if args.use_metadata else None
    )
    
    # Create data generators
    print(f"Creating data generators for fold {args.fold_idx}...")
    image_size = (args.image_size, args.image_size)
    train_generator, val_generator, diagnosis_to_idx, n_classes, n_folds, class_weights = create_efficientnet_generators(
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
        image_size=image_size
    )
    
    # Build the model with multi-label output
    print("Building model...")
    if args.model_size == 'ensemble':
        print("Creating ensemble of EfficientNet models...")
        models = create_efficientnet_ensemble(
            n_classes, 
            input_shape=(*image_size, 3), 
            multi_label=True
        )
        if not models:
            print("Failed to create ensemble. Exiting.")
            return
        model = models[0]  # Use the first model for initial training
    else:
        with tf_strategy.scope():
            model = build_efficientnet_model(
                n_classes, 
                model_size=args.model_size,
                input_shape=(*image_size, 3),
                multi_label=True
            )
    
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
        
        if history is None:
            print("Training failed. Exiting.")
            return
        
        # Save the model
        try:
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
        except Exception as save_error:
            print(f"Error saving model: {save_error}")
        
        # Log initial model to MLflow
        if args.log_to_mlflow:
            try:
                log_model_to_mlflow(model, history, f"efficientnet_{args.model_size}", args.fold_idx, diagnosis_to_idx)
            except Exception as mlflow_error:
                print(f"Error logging to MLflow: {mlflow_error}")
        
        # Fine-tune if requested
        if args.fine_tune:
            print("Fine-tuning model...")
            try:
                # Fine-tune the model
                with tf_strategy.scope():
                    model = fine_tune_efficientnet(model, learning_rate=1e-5, fine_tune_layers=100)
                
                fine_tune_history = train_model(
                    model,
                    train_generator,
                    val_generator,
                    args.fine_tune_epochs,
                    args.early_stopping,
                    class_weights=class_weights,
                    learning_rate=1e-5,
                    multi_label=True,
                    batch_size=args.batch_size
                )
                
                if fine_tune_history is not None:
                    history = fine_tune_history
                
                    # Save the fine-tuned model
                    try:
                        model.save(model_save_path.replace('.keras', '_fine_tuned.keras'))
                        print(f"Fine-tuned model saved")
                    except Exception as save_error:
                        print(f"Error saving fine-tuned model: {save_error}")
                    
                    if args.log_to_mlflow:
                        try:
                            log_model_to_mlflow(model, fine_tune_history, f"fine_tuned_efficientnet_{args.model_size}", args.fold_idx, diagnosis_to_idx)
                        except Exception as mlflow_error:
                            print(f"Error logging fine-tuned model to MLflow: {mlflow_error}")
            except Exception as fine_tune_error:
                print(f"Error during fine-tuning: {fine_tune_error}")
        
        # Evaluate the model
        print("Evaluating model...")
        try:
            evaluation_results = evaluate_model(
                model, 
                val_generator, 
                class_indices=diagnosis_to_idx,
                save_path="models/evaluation"
            )
            
            print("\nEvaluation Results:")
            for metric, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
            
            # Log evaluation results to MLflow
            if args.log_to_mlflow:
                with mlflow.start_run(run_name=f"evaluation_efficientnet_{args.model_size}_fold_{args.fold_idx}"):
                    for metric, value in evaluation_results.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(metric, value)
        
        except Exception as eval_error:
            print(f"Error during evaluation: {eval_error}")
    
    except Exception as train_error:
        print(f"Error during training: {train_error}")

if __name__ == "__main__":
    main()