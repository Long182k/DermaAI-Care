import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
import multiprocessing as mp
from src.custom_layers import CustomScaleLayer
from tensorflow.keras.models import load_model
from src.data_preprocessing import create_generators, analyze_dataset
from src.model_training import build_model, train_model, fine_tune_model, strategy
from src.evaluate_model import evaluate_model, save_model

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
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
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
    # Setup GPU first
    setup_gpu()
    
    # Define paths
    CSV_PATH = "data/ISIC_2020_Training_GroundTruth_v2.csv"
    IMAGE_DIR = "data/train"
    MODEL_SAVE_PATH = "models/skin_disease_model.h5"
    CHECKPOINT_PATH = "models/checkpoint.h5"  # Path to load pre-trained model
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Analyze dataset
        print("Analyzing dataset...")
        dataset_stats = analyze_dataset(CSV_PATH)
        
        # Create data generators with smaller batch size
        train_gen, val_gen, diagnosis_to_idx, n_classes, _ = create_generators(
            CSV_PATH,
            IMAGE_DIR,
            batch_size=16,
            min_samples_per_class=10
        )
        
        # Choose whether to start from Phase 1 or load model for Phase 2
        START_FROM_PHASE_2 = True  # Set this to True to start from Phase 2
        
        if not START_FROM_PHASE_2:
            # Phase 1: Initial Training
            print("Phase 1: Initial Training...")
            model = build_model(num_classes=n_classes)
            history1 = train_model(
                model, 
                train_gen, 
                val_gen,
                epochs=8,
                class_weights=dataset_stats['class_weights'],
                early_stopping_patience=3,
                reduce_lr_patience=2
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
                    print("Creating fresh model...")
                    # Create a new model if loading fails
                    model = build_model(num_classes=n_classes)
                    print("Fresh model created successfully.")
        
        # Clear session before fine-tuning
        tf.keras.backend.clear_session()
        
        print("Phase 2: Fine-tuning...")
        history2 = fine_tune_model(
            model,
            train_gen,
            val_gen,
            epochs=5,
            early_stopping_patience=2
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