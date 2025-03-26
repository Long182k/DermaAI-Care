import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
import multiprocessing as mp
import numpy as np
from src.custom_layers import CustomScaleLayer
from tensorflow.keras.models import load_model
from src.data_preprocessing import create_yolo_generators, analyze_yolo_dataset
from src.model_training import build_peft_model, train_model, fine_tune_model, strategy, LoRALayer
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
        
        # Set environment variables for GPU detection
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Configure TF to use GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s): {gpus}")
            for gpu in gpus:
                try:
                    # Memory growth needs to be set before GPUs have been initialized
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for {gpu}")
                except RuntimeError as e:
                    print(f"Error configuring GPU {gpu}: {e}")
            
            # Verify GPU is available
            print("TensorFlow GPU configuration:")
            print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
            
            # Print device placement for a simple operation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"Matrix multiplication result: {c}")
                print(f"Tensor device: {c.device}")
                
            print("GPU setup successful!")
        else:
            print("No GPU devices found by TensorFlow. Running on CPU")
            print("Please run the setup_gpu.py script to diagnose and fix GPU issues")
            
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
    CSV_PATH = "/Users/drake/Documents/UWE/DermaAI-Care/Training/data/ISIC_2020_Training_GroundTruth_v2.csv"
    IMAGE_DIR = "/Users/drake/Documents/UWE/DermaAI-Care/Training/image_processing/yolov5/runs/detect/exp"
    LABELS_DIR = "/Users/drake/Documents/UWE/DermaAI-Care/Training/image_processing/yolov5/runs/detect/exp/labels"
    MODEL_SAVE_PATH = "models/skin_cancer_prediction_model.keras"
    CHECKPOINT_PATH = "models/checkpoint.keras"  # Path to load pre-trained model
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Analyze dataset
        print("Analyzing dataset...")
        dataset_stats = analyze_yolo_dataset(CSV_PATH, IMAGE_DIR, LABELS_DIR)
        
        # Get dataset size to adjust parameters
        dataset_size = len(dataset_stats.get('class_weights', {}))
        print(f"Dataset size: {dataset_size} classes")
        
        # Adjust parameters based on dataset size
        if dataset_size < 1000:  # Small dataset (e.g., 100 images)
            batch_size = 32
            epochs_phase1 = 20
            epochs_phase2 = 10
            early_stopping_patience = 5
            reduce_lr_patience = 3
            lora_r = 8
        else:  # Large dataset (e.g., 33K images)
            batch_size = 64
            epochs_phase1 = 30
            epochs_phase2 = 15
            early_stopping_patience = 8
            reduce_lr_patience = 4
            lora_r = 16
        
        print(f"Using batch size: {batch_size}, epochs: {epochs_phase1}/{epochs_phase2}")
        
        # Number of folds for cross-validation
        n_folds = 5  # Reduced from 10 to speed up training
        
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
            train_gen, val_gen, diagnosis_to_idx, n_classes, _ = create_yolo_generators(
                CSV_PATH,
                IMAGE_DIR,
                LABELS_DIR,
                batch_size=batch_size,  # Use dynamic batch size
                min_samples_per_class=5,
                n_folds=n_folds,
                fold_idx=fold_idx
            )
            
            # Choose whether to start from Phase 1 or load model for Phase 2
            START_FROM_PHASE_2 = False  # Set this to True to start from Phase 2
            
            if not START_FROM_PHASE_2:
                # Phase 1: Initial training with frozen base model
                print("\nPhase 1: Initial training with PEFT...")
                
                # Build model with PEFT optimization
                model = build_peft_model(n_classes, r=lora_r, alpha=32)
                
                # Print model summary
                model.summary()
                
                # Train the model
                history = train_model(
                    model,
                    train_gen,
                    val_gen,
                    epochs=epochs_phase1,  # Use dynamic epochs
                    early_stopping_patience=early_stopping_patience,
                    reduce_lr_patience=reduce_lr_patience,
                    class_weights=dataset_stats['class_weights']
                )
            else:
                # Load pre-trained model for Phase 2
                print(f"\nLoading pre-trained model from {CHECKPOINT_PATH}...")
                try:
                    # Register custom objects
                    custom_objects = {
                        'CustomScaleLayer': CustomScaleLayer,
                        'LoRALayer': LoRALayer
                    }
                    model = load_model(CHECKPOINT_PATH, custom_objects=custom_objects)
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Building new model instead...")
                    model = build_peft_model(n_classes, r=8, alpha=32)
            
            # Phase 2: Fine-tuning
            print("\nPhase 2: Fine-tuning...")
            history_ft = fine_tune_model(
                model,
                train_gen,
                val_gen,
                epochs=epochs_phase2,  # Use dynamic epochs for fine-tuning
                early_stopping_patience=early_stopping_patience
            )
            
            # Evaluate the model
            print("\nEvaluating model...")
            metrics = evaluate_model(model, val_gen)
            
            # Store metrics for this fold
            for metric_name, value in metrics.items():
                if metric_name in all_metrics:
                    all_metrics[metric_name].append(value)
            
            # Save the model for this fold
            fold_model_path = f"models/skin_cancer_model_fold_{fold_idx+1}.keras"
            save_model(model, fold_model_path)
            
            # Clear memory
            del model, train_gen, val_gen
            tf.keras.backend.clear_session()
        
        # Calculate and print average metrics across all folds
        print("\n" + "="*50)
        print("Cross-validation complete")
        print("Average metrics across all folds:")
        for metric_name, values in all_metrics.items():
            if values:  # Check if the list is not empty
                avg_value = np.mean(values)
                std_value = np.std(values)
                print(f"{metric_name}: {avg_value:.4f} ± {std_value:.4f}")
        
        # Train final model on all data
        print("\n" + "="*50)
        print("Training final model on all data...")
        
        # Create generators using all data
        all_data_df, _, diagnosis_to_idx = load_yolo_detections(
            IMAGE_DIR,
            LABELS_DIR,
            CSV_PATH,
            min_samples_per_class=5,
            n_folds=n_folds
        )
        
        all_data_gen = YOLODetectionGenerator(
            all_data_df,
            diagnosis_to_idx,
            batch_size=64,
            is_training=True
        )
        
        # Build final model
        final_model = build_peft_model(len(diagnosis_to_idx), r=8, alpha=32)
        
        # Train final model
        final_history = train_model(
            final_model,
            all_data_gen,
            None,  # No validation data for final training
            epochs=40,
            early_stopping_patience=15,
            reduce_lr_patience=8,
            class_weights=dataset_stats['class_weights']
        )
        
        # Fine-tune final model
        final_history_ft = fine_tune_model(
            final_model,
            all_data_gen,
            None,  # No validation data for final training
            epochs=25,
            early_stopping_patience=10
        )
        
        # Save final model
        save_model(final_model, MODEL_SAVE_PATH)
        print(f"Final model saved to {MODEL_SAVE_PATH}")
        
    except Exception as e:
        import traceback
        print(f"Error in training: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()