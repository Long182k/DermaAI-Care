import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
import multiprocessing as mp
import numpy as np

# Set multiprocessing start method to 'spawn'
if __name__ == '__main__':
    # Must be called before any other multiprocessing code
    mp.set_start_method('spawn', force=True)

def setup_gpu():
    try:
        # Reset GPU devices
        tf.keras.backend.clear_session()
        
        # Configure TF to use GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                try:
                    # Memory growth needs to be set before GPUs have been initialized
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Reduce memory limit further to avoid OOM
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]  # Reduced from 3072
                    )
                    print(f"Successfully configured GPU: {gpu}")
                except RuntimeError as e:
                    print(f"Error configuring GPU {gpu}: {e}")
            
            # Additional memory optimization settings
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '1'
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
            
            # Set mixed precision policy
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Verify GPU is available
            if tf.test.is_built_with_cuda():
                print("CUDA is available")
                print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            else:
                print("WARNING: CUDA is not available")
        else:
            print("No GPU devices found. Running on CPU")
            
    except Exception as e:
        print(f"Error setting up GPU: {e}")
        print("Falling back to CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

from src.data_preprocessing import create_generators, analyze_dataset
from src.model_training import build_model, train_model, fine_tune_model
from src.evaluate_model import evaluate_model, save_model

def main():
    # Setup GPU first
    setup_gpu()
    
    # Define paths
    CSV_PATH = "data/ISIC_2020_Training_GroundTruth_v2.csv"
    IMAGE_DIR = "data/train"
    MODEL_SAVE_PATH = "models/skin_disease_model_{fold}.h5"  # Add fold number to filename
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Analyze dataset
        print("Analyzing dataset...")
        dataset_stats = analyze_dataset(CSV_PATH)
        
        # Parameters for k-fold cross validation
        n_folds = 5  # Number of folds
        batch_size = 16  # Reduced batch size for Colab
        
        # Store metrics for each fold
        fold_metrics = []
        
        # Train model for each fold
        for fold in range(n_folds):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold + 1}/{n_folds}")
            print(f"{'='*50}")
            
            # Clear session at the start of each fold
            tf.keras.backend.clear_session()
            
            # Create data generators for this fold
            train_gen, val_gen, diagnosis_to_idx, n_classes, _ = create_generators(
                CSV_PATH,
                IMAGE_DIR,
                fold_idx=fold,
                batch_size=batch_size,
                min_samples_per_class=10,
                n_folds=n_folds
            )
            
            # Build model
            model = build_model(num_classes=n_classes)
            
            # Phase 1: Initial Training with frozen layers
            print(f"\nPhase 1: Initial Training Fold {fold + 1}...")
            history1 = train_model(
                model,
                train_gen,
                val_gen,
                epochs=15,  # More epochs for initial training
                class_weights=dataset_stats['class_weights']  # Use class weights for imbalanced data
            )
            
            # Phase 2: Fine-tuning
            print(f"\nPhase 2: Fine-tuning Fold {fold + 1}...")
            history2 = fine_tune_model(
                model,
                train_gen,
                val_gen,
                epochs=10
            )
            
            # Evaluate model for this fold
            print(f"\nEvaluating Fold {fold + 1}...")
            metrics = evaluate_model(model, val_gen)
            fold_metrics.append(metrics)
            
            # Save model for this fold
            fold_model_path = MODEL_SAVE_PATH.format(fold=fold + 1)
            print(f"\nSaving model for Fold {fold + 1}...")
            save_model(model, fold_model_path)
            
            # Clear memory
            tf.keras.backend.clear_session()
            del model
            del train_gen
            del val_gen
        
        # Print average metrics across all folds
        print("\nAverage Metrics Across All Folds:")
        print("-" * 50)
        metrics_keys = fold_metrics[0].keys()
        for key in metrics_keys:
            avg_value = sum(fold[key] for fold in fold_metrics) / len(fold_metrics)
            std_value = np.std([fold[key] for fold in fold_metrics])
            print(f"{key}: {avg_value:.4f} ± {std_value:.4f}")
        
        print("\nTraining complete! Models saved in models/ directory")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 