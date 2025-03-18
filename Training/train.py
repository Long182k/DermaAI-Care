import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
import multiprocessing as mp
import numpy as np
from src.custom_layers import CustomScaleLayer
from tensorflow.keras.models import load_model
from src.data_preprocessing import create_generators, analyze_dataset
from src.model_training import build_model, train_model, fine_tune_model, strategy
from src.evaluate_model import evaluate_model, save_model
from src.utils import set_global_seeds

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
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
                    )
                    print(f"Successfully configured GPU: {gpu}")
                except RuntimeError as e:
                    print(f"Error configuring GPU {gpu}: {e}")
            
            # Additional memory optimization settings
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '1'
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
            
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

def main():
    # Set global random seeds for reproducibility
    set_global_seeds(42)
    
    # Setup GPU first
    setup_gpu()
    
    # Define paths
    CSV_PATH = "data/ISIC_2020_Training_GroundTruth_v2.csv"
    IMAGE_DIR = "data/train"
    MODEL_SAVE_PATH = "models/skinning_cancer_prediction_model.h5"
    CHECKPOINT_PATH = "models/checkpoint.h5"  # Path to load pre-trained model
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Analyze dataset
        print("Analyzing dataset...")
        dataset_stats = analyze_dataset(CSV_PATH)
        
        # Number of folds for cross-validation
        n_folds = 10
        
        # Store metrics for each fold
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'sensitivity': [],
            'f1': [],
            'auc': [],
            'specificity': [],
            'icbhi_score': []
        }
        
        # Perform k-fold cross-validation
        for fold_idx in range(n_folds):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold_idx+1}/{n_folds}")
            print(f"{'='*50}")
            
            # Clear session before each fold
            tf.keras.backend.clear_session()
            
            # Create data generators for this fold
            train_gen, val_gen, diagnosis_to_idx, n_classes, _ = create_generators(
                CSV_PATH,
                IMAGE_DIR,
                batch_size=24,
                min_samples_per_class=10,
                n_folds=n_folds,
                fold_idx=fold_idx
            )
            
            # Choose whether to start from Phase 1 or load model for Phase 2
            START_FROM_PHASE_2 = False  # Set this to True to start from Phase 2
            
            if not START_FROM_PHASE_2:
                # Phase 1: Initial Training
                print("Phase 1: Initial Training...")
                model = build_model(num_classes=n_classes)
                history1 = train_model(
                    model, 
                    train_gen, 
                    val_gen,
                    epochs=2,
                    # class_weights=dataset_stats['class_weights'],
                    early_stopping_patience=3,
                    reduce_lr_patience=3,
                    class_weights=None
                )
            else:
                # Load pre-trained model for Phase 2
                print("Loading pre-trained model for Phase 2...")
                # Define custom objects dictionary
                custom_objects = {
                    'CustomScaleLayer': CustomScaleLayer,
                    'accuracy': tf.keras.metrics.Accuracy(),
                    'auc': tf.keras.metrics.AUC(),
                    'precision': tf.keras.metrics.Precision(),
                    'recall': tf.keras.metrics.Recall()
                }
                
                # Load the model with custom objects within strategy scope
                with strategy.scope():
                    try:
                        model = load_model(CHECKPOINT_PATH, custom_objects=custom_objects)
                        print("Model loaded successfully!")
                    except Exception as e:
                        print(f"Error loading saved model: {e}")
                        continue  # Skip to next fold instead of returning
            
            # Clear session before fine-tuning
            tf.keras.backend.clear_session()
            
            print("Phase 2: Fine-tuning...")
            history2 = fine_tune_model(
                model,
                train_gen,
                val_gen,
                epochs=1,
                early_stopping_patience=4
            )
            
            # Evaluate model
            print(f"Evaluating model for fold {fold_idx+1}...")
            fold_metrics = evaluate_model(model, val_gen)
            
            # Save fold metrics
            for metric, value in fold_metrics.items():
                all_metrics[metric].append(value)
            
            # Save model for this fold
            fold_model_path = f"models/fold_{fold_idx+1}_model.h5"
            save_model(model, fold_model_path)
            print(f"Model for fold {fold_idx+1} saved to {fold_model_path}")
        
        # Calculate and print cross-validation summary
        print("\n" + "="*50)
        print("10-Fold Cross-Validation Summary:")
        print("="*50)
        
        for metric in all_metrics:
            values = all_metrics[metric]
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric.capitalize()} - Mean: {mean_value:.4f}, Std: {std_value:.4f}")
        
        # Save the final model (could be an ensemble or the best model)
        # For simplicity, we'll save the last fold's model as the final model
        save_model(model, MODEL_SAVE_PATH)
        
        print(f"\nTraining complete! Final model saved to {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()