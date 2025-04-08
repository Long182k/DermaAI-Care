import tensorflow as tf
import numpy as np
import os
import traceback
import gc

def train_model(model, train_generator, val_generator, epochs=50, batch_size=32, 
                early_stopping_patience=10, reduce_lr_patience=5, 
                callbacks=None, class_weights=None, model_save_path=None,
                learning_rate=0.0001, multi_label=False):
    """
    Train the model without using workers or multiprocessing
    """
    try:
        print(f"Training with batch size: {batch_size}")
        
        # Create model checkpoint directory
        if model_save_path:
            model_dir = os.path.dirname(model_save_path)
        else:
            model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, 'checkpoint.keras')
        
        # If callbacks weren't provided, create them
        if callbacks is None:
            # Set up callbacks with better learning rate scheduling
            monitor_metric = 'val_loss'
            mode = 'min'
            
            callbacks = [
                # Early stopping to prevent overfitting
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    mode=mode,
                    verbose=1
                ),
                # Reduce learning rate when training plateaus
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=0.2,
                    patience=reduce_lr_patience,
                    min_lr=1e-7,
                    mode=mode,
                    verbose=1
                ),
                # Save best model checkpoint
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=monitor_metric,
                    save_best_only=True,
                    mode=mode,
                    verbose=1
                )
            ]
        
        # Train the model - WORKERS AND USE_MULTIPROCESSING REMOVED
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        return None

def fine_tune_model(model, train_generator, val_generator, epochs=5, early_stopping_patience=3, 
                   multi_label=True, class_weights=None, batch_size=32, model_save_path=None):
    """
    Fine-tune the model without using workers or multiprocessing
    """
    try:
        # First, verify the base model's performance before fine-tuning
        print("Evaluating base model performance before fine-tuning...")
        base_eval = model.evaluate(val_generator, verbose=1)
        base_metrics = dict(zip(model.metrics_names, base_eval))
        
        if 'accuracy' in base_metrics and base_metrics['accuracy'] < 0.5:
            print(f"WARNING: Base model accuracy ({base_metrics['accuracy']:.4f}) is too low for fine-tuning.")
            print("Fine-tuning may not improve performance. Consider retraining with a different approach.")
        
        # Make a clone of the model
        fine_tuned_model = tf.keras.models.clone_model(model)
        fine_tuned_model.set_weights(model.get_weights())
        
        print("Creating fine-tuning model by selectively unfreezing layers...")
        
        # Configure the layers for fine-tuning
        # Unfreeze only the dense layers, not the final output layer
        fine_tuned_layers = []
        for layer in fine_tuned_model.layers:
            # Start with everything frozen
            layer.trainable = False
            
            # Unfreeze only the dense layers before the final layer
            if isinstance(layer, tf.keras.layers.Dense):
                if layer != fine_tuned_model.layers[-1]:  # Not the output layer
                    layer.trainable = True
                    fine_tuned_layers.append(layer.name)
        
        print(f"Layers unfrozen for fine-tuning: {fine_tuned_layers}")
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=early_stopping_patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ]
        
        # Compile model with very small learning rate
        fine_tuned_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=model.loss,
            metrics=model.metrics
        )
        
        # Ensure class balance is preserved
        if class_weights is None:
            print("WARNING: No class weights provided for fine-tuning. This may lead to class imbalance issues.")
        else:
            print(f"Using class weights for fine-tuning: {class_weights}")
        
        # Train with a smaller batch size for stability
        fine_tune_batch_size = min(16, batch_size)
        
        try:
            # WORKERS AND USE_MULTIPROCESSING REMOVED
            history = fine_tuned_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                batch_size=fine_tune_batch_size,
                verbose=1
            )
        except Exception as fit_error:
            print(f"Error during fine-tuning fit: {fit_error}")
            traceback.print_exc()
            return None, model  # Return original model on error
        
        # Save the fine-tuned model if path provided
        if model_save_path:
            fine_tuned_path = model_save_path.replace('.keras', '_fine_tuned.keras')
            fine_tuned_model.save(fine_tuned_path)
            print(f"Fine-tuned model saved to {fine_tuned_path}")
        
        return history, fine_tuned_model
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        traceback.print_exc()
        return None, model  # Return the original model in case of error

# Add this function to help import in train.py
def get_fixed_train_functions():
    """Returns the fixed training functions"""
    return train_model, fine_tune_model 