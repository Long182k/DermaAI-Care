import os
import argparse
import tensorflow as tf
import numpy as np
import gc
import traceback

# Import custom modules
from src.data_preprocessing import create_yolo_generators, analyze_yolo_dataset, set_random_seeds
from src.model_training import build_peft_model, setup_gpu, log_model_to_mlflow, TimeoutCallback
from src.model_training_fixed import train_model, fine_tune_model  # Import the fixed functions
from src.evaluate_model import evaluate_model

# Define model save paths
MODEL_SAVE_PATH = "/kaggle/working/DermaAI-Care/Training/training_model/models/skin_cancer_prediction_model.keras"

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train skin lesion classification model')
    parser.add_argument('--csv_path', type=str, default='/kaggle/input/annotated-isic-2019-images/ISIC_2019_Training_GroundTruth.csv',
                        help='Path to CSV file with image metadata')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/annotated-isic-2019-images/exp/exp',
                        help='Directory containing detected images')
    parser.add_argument('--labels_dir', type=str, default='/kaggle/input/annotated-isic-2019-images/labels/labels',
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--augmentation_strength', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Strength of data augmentation')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--multi_label', action='store_true', default=True,
                        help='Use multi-label classification')
    parser.add_argument('--metadata_csv_path', type=str, default="/kaggle/input/annotated-isic-2019-images/ISIC_2019_Training_Metadata.csv",
                       help='Path to CSV file with patient metadata (age, sex, anatomical site)')
    parser.add_argument('--use_metadata', action='store_true',
                       help='Whether to use patient metadata (age, sex, anatomical site)')
    
    args = parser.parse_args()
    
    try:
        # Set random seeds for reproducibility
        set_random_seeds(args.seed)
        
        # Clear any existing session to avoid conflicts
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Update model save path if provided
        model_save_path = MODEL_SAVE_PATH
        if args.model_save_path:
            model_save_path = args.model_save_path
            print(f"Custom model save path set: {model_save_path}")
        
        # Create model save directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Analyze dataset
        print("Analyzing dataset...")
        analyze_yolo_dataset(
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
        
        # Build the model
        print("Building model...")
        model = build_peft_model(n_classes, multi_label=args.multi_label)
        
        if model is None:
            print("Failed to build model. Exiting.")
            return
        
        # Print model summary
        model.summary()
        
        # Train the model
        print("Training model...")
        try:
            # Configure timeout callback
            timeout_seconds = 14400  # 4 hours
            callbacks = [
                TimeoutCallback(timeout_seconds=timeout_seconds)
            ]
            
            # Use the fixed train_model function without workers/use_multiprocessing
            history = train_model(
                model=model,
                train_generator=train_generator,
                val_generator=val_generator,
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping,
                reduce_lr_patience=args.reduce_lr,
                callbacks=callbacks,
                class_weights=class_weights,
                learning_rate=args.learning_rate,
                multi_label=args.multi_label,
                batch_size=args.batch_size,
                model_save_path=model_save_path
            )
            
            # Save the model
            try:
                model.save(model_save_path)
                print(f"Model saved to {model_save_path}")
            except Exception as save_error:
                print(f"Error saving model: {save_error}")
                traceback.print_exc()
        
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
        
        # Fine-tune the model if requested
        if args.fine_tune:
            print("Fine-tuning model...")
            try:
                # Use the fixed fine_tune_model function without workers/use_multiprocessing
                fine_tune_history, fine_tuned_model = fine_tune_model(
                    model=model,
                    train_generator=train_generator,
                    val_generator=val_generator,
                    epochs=args.fine_tune_epochs,
                    early_stopping_patience=args.early_stopping,
                    multi_label=args.multi_label,
                    class_weights=class_weights,
                    batch_size=args.batch_size,
                    model_save_path=model_save_path
                )
                
                if fine_tuned_model is not None:
                    model = fine_tuned_model  # Use the fine-tuned model
            except Exception as fine_tune_error:
                print(f"Error during fine-tuning: {fine_tune_error}")
                traceback.print_exc()
        
        # Evaluate model
        print("\nEvaluating model...")
        try:
            # Ensure evaluation directory exists
            eval_dir = os.path.join(os.path.dirname(model_save_path), "evaluation_results")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Get the class names from the diagnosis_to_idx mapping
            class_names = list(diagnosis_to_idx.keys())
            
            # Print class mapping for debug purposes
            print("Class mapping for evaluation:")
            for i, class_name in enumerate(class_names):
                print(f"  {i}: {class_name}")
            
            metrics = evaluate_model(
                model=model,
                test_generator=val_generator,
                class_names=class_names,
                output_dir=eval_dir
            )
            
            if metrics:
                print("\nEvaluation Metrics:")
                print("-" * 55)
                print(f"Validation Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"Validation Precision: {metrics.get('precision', 0):.4f}")
                print(f"Validation Recall/Sensitivity: {metrics.get('recall', 0):.4f}")
                print(f"Validation F1 Score: {metrics.get('f1', 0):.4f}")
                print(f"Validation AUC: {metrics.get('auc', 0):.4f}")
                print(f"Validation Specificity: {metrics.get('specificity', 0):.4f}")
                
                # Print paths of saved files
                if 'confusion_matrix_path' in metrics:
                    print(f"\nConfusion matrix saved to: {metrics['confusion_matrix_path']}")
                if 'report_path' in metrics:
                    print(f"Classification report saved to: {metrics['report_path']}")
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in main training process: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 