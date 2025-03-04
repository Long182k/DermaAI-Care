import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
import multiprocessing as mp

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
    MODEL_SAVE_PATH = "models/skin_disease_model.h5"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Analyze dataset
        print("Analyzing dataset...")
        dataset_stats = analyze_dataset(CSV_PATH)
        
        # Create data generators with smaller batch size
        train_gen, val_gen, diagnosis_to_idx, n_classes = create_generators(
            CSV_PATH,
            IMAGE_DIR,
            batch_size=16,  # Reduced from 32
            min_samples_per_class=10
        )
        
        # Build model with memory optimizations
        model = build_model(num_classes=n_classes)
        
        # Use mixed precision for better memory efficiency
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        history1 = train_model(
            model, 
            train_gen, 
            val_gen,
            epochs=10,  # Reduced from 15
            class_weights=dataset_stats['class_weights']
        )
        
        print("Phase 2: Fine-tuning...")
        history2 = fine_tune_model(
            model,
            train_gen,
            val_gen,
            epochs=10
        )
        
        # Evaluate model
        print("Evaluating model...")
        evaluate_model(model, val_gen)
        
        # Save model
        print("Saving model...")
        save_model(model, MODEL_SAVE_PATH)
        
        print(f"Training complete! Model saved to {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 