from mlflow.models.signature import infer_signature
from PIL import Image
from sklearn.model_selection import KFold
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Layer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.models import clone_model
from tensorflow.keras.metrics import Mean, BinaryAccuracy, AUC, Precision, Recall
import numpy as np
import os
import time
import gc
import traceback
import matplotlib.pyplot as plt
import json

# Import PEFT from Hugging Face
try:
    from peft import (
        get_peft_model, 
        LoraConfig, 
        TaskType, 
        PeftModel, 
        PeftConfig
    )
    PEFT_AVAILABLE = True
except ImportError:
    print("PEFT library not found. Installing...")
    os.system("pip install -q peft")
    try:
        from peft import (
            get_peft_model, 
            LoraConfig, 
            TaskType, 
            PeftModel, 
            PeftConfig
        )
        PEFT_AVAILABLE = True
    except ImportError:
        print("Failed to install PEFT. Using standard fine-tuning instead.")
        PEFT_AVAILABLE = False

# Define strategy at module level so it can be imported
# Use CPU strategy by default, will be updated in setup_gpu
strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

# Custom LoRA layer for TensorFlow
class LoRALayer(tf.keras.layers.Layer):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning
    """
    def __init__(self, in_features, out_features, r=8, alpha=32, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        
    def build(self, input_shape):
        # Initialize LoRA matrices
        self.lora_A = self.add_weight(
            name="lora_A",
            shape=(self.in_features, self.r),
            initializer="random_normal",
            trainable=True
        )
        
        self.lora_B = self.add_weight(
            name="lora_B",
            shape=(self.r, self.out_features),
            initializer="zeros",
            trainable=True
        )
        
        # Build the layer
        super(LoRALayer, self).build(input_shape)
        
    def call(self, inputs):
        # LoRA adaptation: inputs @ (A @ B) * (alpha / r)
        lora_output = tf.matmul(inputs, tf.matmul(self.lora_A, self.lora_B)) * (self.alpha / self.r)
        return lora_output
        
    def get_config(self):
        config = super(LoRALayer, self).get_config()
        config.update({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "r": self.r,
            "alpha": self.alpha
        })
        return config

# Add a custom focal loss function to better handle class imbalance
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        gamma: Focusing parameter that reduces the loss contribution from easy examples
        alpha: Weighting factor for the rare class
        
    Returns:
        A focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent NaN losses
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        
        # Sum over classes
        return tf.reduce_sum(loss, axis=-1)
    
    return focal_loss_fixed

# Modify the model compilation in build_peft_model function
def build_peft_model(num_classes, multi_label=False):
    """
    Build a PEFT (Parameter-Efficient Fine-Tuning) model based on EfficientNetB0
    
    Args:
        num_classes: Number of output classes (must be a positive integer)
        multi_label: Whether to use multi-label classification (sigmoid activation)
    
    Returns:
        Compiled Keras model
    """
    try:
        # Validate num_classes
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        
        # Clear any existing session
        tf.keras.backend.clear_session()
        
        # Set data format to channels_last (NHWC) explicitly
        tf.keras.backend.set_image_data_format('channels_last')
        
        # Memory optimization
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Don't pre-allocate memory; allocate as-needed
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                # Memory growth must be set before GPUs have been initialized
                print("Memory growth already set")
        
        # Create base model with proper configuration and weights
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the model using Functional API
        inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')
        
        # Explicitly apply preprocessing to ensure format consistency
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Pass through the base model
        x = base_model(x, training=False)
        
        # Add classification head with proper shape
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Use sigmoid for multi-label, softmax for single-label
        activation = 'sigmoid' if multi_label else 'softmax'
        outputs = tf.keras.layers.Dense(num_classes, activation=activation, name='output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use mixed precision for better performance
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Define metrics
        metrics = [
            tf.keras.metrics.AUC(multi_label=multi_label, name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        
        if multi_label:
            metrics.append(tf.keras.metrics.BinaryAccuracy(name='accuracy'))
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            metrics.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        
        # Compile the model with appropriate optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss,
            metrics=metrics
        )
        
        return model
        
    except Exception as e:
        print(f"Error building model: {e}")
        traceback.print_exc()
        return None

# Custom callbacks for memory management and timeout protection
class TimeoutCallback(tf.keras.callbacks.Callback):
    """
    Callback to stop training after a certain amount of time has passed.
    
    Args:
        timeout_seconds: Number of seconds after which training will be stopped
        checkpoint_path: Path to save the model before timeout
    """
    def __init__(self, timeout_seconds=12000, checkpoint_path=None):
        super(TimeoutCallback, self).__init__()
        self.timeout_seconds = timeout_seconds
        self.checkpoint_path = checkpoint_path
        self.start_time = time.time()
        print(f"Training will timeout after {timeout_seconds/3600:.1f} hours")
        
    def on_epoch_begin(self, epoch, logs=None):
        # Reset timer at the beginning of each epoch
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        # Check if we've exceeded the timeout
        elapsed_time = time.time() - self.start_time
        epoch_time = time.time() - self.epoch_start_time
        
        # Print time info
        print(f"Epoch {epoch} took {epoch_time:.1f}s. Total elapsed time: {elapsed_time/60:.1f} min")
        
        # Save checkpoint if needed
        if self.checkpoint_path and elapsed_time > 0.7 * self.timeout_seconds:
            print(f"Saving checkpoint before potential timeout at {self.checkpoint_path}")
            self.model.save(self.checkpoint_path)
            
        if elapsed_time > self.timeout_seconds:
            self.model.stop_training = True
            print(f"\nStopping training due to timeout ({self.timeout_seconds} seconds).")
            
            # Save the final model state
            if self.checkpoint_path:
                print(f"Saving final model at {self.checkpoint_path}")
                self.model.save(self.checkpoint_path)
                
                # Save training state for resumption
                self.save_training_state(epoch, logs)
    
    def save_training_state(self, epoch, logs):
        """Save the training state for later resumption"""
        if not self.checkpoint_path:
            return
            
        resume_info = {
            'epoch': epoch + 1,  # Next epoch to resume from
            'optimizer_config': self.model.optimizer.get_config(),
            'last_logs': logs,
            'elapsed_time': time.time() - self.start_time
        }
        
        # Save as JSON next to the model
        resume_path = os.path.join(os.path.dirname(self.checkpoint_path), 'resume_info.json')
        with open(resume_path, 'w') as f:
            # Convert numpy values to Python native types
            logs_serializable = {}
            for k, v in logs.items():
                if isinstance(v, (np.float32, np.float64)):
                    logs_serializable[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    logs_serializable[k] = int(v)
                else:
                    logs_serializable[k] = v
            
            resume_info['last_logs'] = logs_serializable
            json.dump(resume_info, f)
        
        print(f"Saved training state to {resume_path}")

class TrainingResumer:
    """
    Helper class to resume training from a previous checkpoint
    
    Args:
        model_path: Path to the saved model
        resume_info_path: Path to the resume info JSON
    """
    def __init__(self, model_path, resume_info_path=None):
        self.model_path = model_path
        self.resume_info_path = resume_info_path or os.path.join(
            os.path.dirname(model_path), 'resume_info.json')
        self.resume_info = None
        
        if os.path.exists(self.resume_info_path):
            try:
                with open(self.resume_info_path, 'r') as f:
                    self.resume_info = json.load(f)
                print(f"Found resume information: starting from epoch {self.resume_info['epoch']}")
            except Exception as e:
                print(f"Error loading resume info: {e}")
                self.resume_info = None
    
    def load_model(self):
        """Load the model from checkpoint"""
        if os.path.exists(self.model_path):
            try:
                model = tf.keras.models.load_model(self.model_path)
                print(f"Loaded model from {self.model_path}")
                return model
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        return None
    
    def get_initial_epoch(self):
        """Get the epoch to resume from"""
        if self.resume_info:
            return self.resume_info.get('epoch', 0)
        return 0
    
    def restore_optimizer_state(self, model):
        """Restore optimizer state if available"""
        if self.resume_info and 'optimizer_config' in self.resume_info:
            try:
                # Create a new optimizer with the saved config
                optimizer_config = self.resume_info['optimizer_config']
                optimizer_class = getattr(tf.keras.optimizers, optimizer_config['name'])
                optimizer = optimizer_class.from_config(optimizer_config)
                
                # Compile the model with the restored optimizer
                model.compile(
                    optimizer=optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
                print("Restored optimizer state")
            except Exception as e:
                print(f"Error restoring optimizer state: {e}")
        return model

class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def __init__(self, cleanup_frequency=5):
        super(MemoryCleanupCallback, self).__init__()
        self.cleanup_frequency = cleanup_frequency
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.cleanup_frequency == 0:
            # Perform memory cleanup
            gc.collect()
            tf.keras.backend.clear_session()
            print("\nPerformed memory cleanup")
            
            # Log memory usage if psutil is available
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
            except ImportError:
                pass

# Update the train_model function signature
def train_model(
    model, 
    train_data, 
    val_data, 
    epochs, 
    early_stopping_patience, 
    reduce_lr_patience, 
    class_weights=None,
    learning_rate=0.0001,
    multi_label=False,
    batch_size=32
):
    """
    Train the model with proper callbacks and memory optimizations
    
    Args:
        model: The compiled Keras model
        train_data: Training data generator or dataset
        val_data: Validation data generator or dataset
        epochs: Number of epochs to train
        early_stopping_patience: Number of epochs to wait before early stopping
        reduce_lr_patience: Number of epochs to wait before reducing learning rate
        class_weights: Dictionary of class weights for imbalanced datasets
        learning_rate: Initial learning rate
        multi_label: Whether this is a multi-label classification task
        batch_size: Batch size for training
    """
    print(f"Training with batch size: {batch_size}")
    
    # Use mixed precision for better performance
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Define model checkpoint directory
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.keras')
    
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
        ),
        # Manage memory during training
        MemoryCleanupCallback(cleanup_frequency=2),
        # Stop training if it takes too long (4 hours)
        TimeoutCallback(timeout_seconds=14400, checkpoint_path=checkpoint_path),
        # Log training metrics
        tf.keras.callbacks.CSVLogger(os.path.join(model_dir, 'training_log.csv'), append=True)
    ]
    
    # Print information about GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Training on {len(gpus)} GPU(s): {gpus}")
        # Try to set memory growth for better memory management
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled on all GPUs")
        except:
            print("Memory growth already set")
    else:
        print("Training on CPU")
    
    # Configure model for training if using eager execution
    if tf.executing_eagerly():
        # Compile with optimized settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=1e-7,
            amsgrad=True
        )
        
        # Choose appropriate loss function
        if multi_label:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        else:
            # Use categorical crossentropy for single-label classification
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        # Compile model with updated settings
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    # Calculate steps per epoch based on dataset size
    if isinstance(train_data, tf.data.Dataset):
        # For tf.data.Dataset, steps per epoch should be None (auto)
        steps_per_epoch = None
        validation_steps = None
    else:
        # For custom generators, ensure we have appropriate steps
        steps_per_epoch = len(train_data)
        validation_steps = len(val_data)
    
    # Train the model with error handling
    try:
        # Perform memory cleanup before training
        gc.collect()
        
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1
        )
        
        # Final cleanup after training
        gc.collect()
        
        return history
    
    except tf.errors.ResourceExhaustedError as oom_error:
        print(f"Out of memory error: {oom_error}")
        print("Attempting to recover by reducing batch size...")
        
        # Try again with reduced batch size
        try:
            reduced_batch = max(1, batch_size // 2)
            print(f"Retrying with batch size: {reduced_batch}")
            
            # Clear existing state
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Reload model from checkpoint if available
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                model = tf.keras.models.load_model(checkpoint_path)
            
            # Train with reduced batch size
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                batch_size=reduced_batch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=1
            )
            
            return history
        
        except Exception as fallback_error:
            print(f"Fallback training also failed: {fallback_error}")
            traceback.print_exc()
            
            # Try to return partial training history if checkpoint exists
            if os.path.exists(checkpoint_path):
                print("Returning partial training results from checkpoint")
                return None
            
            return None
    
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        
        # Try to recover if a checkpoint was saved
        if os.path.exists(checkpoint_path):
            print(f"Training error occurred, but checkpoint was saved at: {checkpoint_path}")
        
        return None

def create_balanced_data_generator(generator, batch_size=32):
    """
    Creates a balanced data generator by oversampling minority classes
    
    Args:
        generator: Original data generator
        batch_size: Batch size for the new generator
        
    Returns:
        A balanced data generator
    """
    try:
        # Get class distribution
        if hasattr(generator, 'classes'):
            classes = generator.classes
            class_indices = generator.class_indices
        else:
            # For custom generators
            classes = []
            for i in range(len(generator)):
                _, batch_y = generator[i]
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    # One-hot encoded
                    batch_classes = np.argmax(batch_y, axis=1)
                else:
                    # Already class indices
                    batch_classes = batch_y
                classes.extend(batch_classes)
            classes = np.array(classes)
            class_indices = {i: i for i in range(len(np.unique(classes)))}
        
        # Count samples per class
        unique_classes = np.unique(classes)
        class_counts = {cls: np.sum(classes == cls) for cls in unique_classes}
        max_samples = max(class_counts.values())
        
        print(f"Original class distribution: {class_counts}")
        
        # Create balanced indices by oversampling minority classes
        balanced_indices = []
        for cls in unique_classes:
            # Get indices for this class
            cls_indices = np.where(classes == cls)[0]
            # Oversample to match the majority class
            if len(cls_indices) < max_samples:
                # Randomly sample with replacement to reach max_samples
                oversampled_indices = np.random.choice(
                    cls_indices, 
                    size=max_samples - len(cls_indices),
                    replace=True
                )
                cls_indices = np.concatenate([cls_indices, oversampled_indices])
            
            balanced_indices.extend(cls_indices)
        
        # Shuffle the indices
        np.random.shuffle(balanced_indices)
        
        # For YOLODetectionGenerator, create a simpler balanced generator
        if isinstance(generator, tf.keras.utils.Sequence) and not hasattr(generator, 'image_shape'):
            # Add image_shape attribute if missing
            if hasattr(generator, 'df') and 'image_path' in generator.df.columns:
                # Try to get image shape from the first image
                try:
                    sample_img = Image.open(generator.df['image_path'].iloc[0])
                    generator.image_shape = sample_img.size + (3,)  # Width, Height, Channels
                except:
                    # Default shape if can't determine
                    generator.image_shape = (224, 224, 3)
            else:
                # Default shape
                generator.image_shape = (224, 224, 3)
        
        # Create a new generator that yields balanced batches
        def balanced_generator():
            while True:
                for i in range(0, len(balanced_indices), batch_size):
                    batch_indices = balanced_indices[i:i + batch_size]
                    
                    # For standard ImageDataGenerator
                    if hasattr(generator, 'filepaths'):
                        batch_x = []
                        batch_y = []
                        
                        for idx in batch_indices:
                            img = generator.image_data_generator.load_img(
                                generator.filepaths[idx],
                                target_size=generator.target_size,
                                color_mode=generator.color_mode
                            )
                            x = generator.image_data_generator.img_to_array(img)
                            x = generator.image_data_generator.standardize(x)
                            
                            batch_x.append(x)
                            batch_y.append(generator.classes[idx])
                        
                        batch_x = np.array(batch_x)
                        batch_y = tf.keras.utils.to_categorical(
                            np.array(batch_y),
                            num_classes=len(generator.class_indices)
                        )
                        
                        yield batch_x, batch_y
                    
                    # For custom generators that support indexing
                    elif hasattr(generator, '__getitem__'):
                        batch_x = []
                        batch_y = []
                        
                        for idx in batch_indices:
                            # Calculate which batch and which item within that batch
                            batch_idx = idx // generator.batch_size
                            item_idx = idx % generator.batch_size
                            
                            # Get the batch
                            if batch_idx < len(generator):
                                x_batch, y_batch = generator[batch_idx]
                                
                                # Check if the item index is valid
                                if item_idx < len(x_batch):
                                    batch_x.append(x_batch[item_idx])
                                    batch_y.append(y_batch[item_idx])
                        
                        if batch_x:  # Only yield if we have data
                            yield np.array(batch_x), np.array(batch_y)
        
        # Create a tf.data.Dataset from the generator
        output_signature = (
            tf.TensorSpec(shape=(None, *generator.image_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(class_indices)), dtype=tf.float32)
        )
        
        balanced_dataset = tf.data.Dataset.from_generator(
            balanced_generator,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)
        
        print(f"Created balanced dataset with equal class distribution")
        return balanced_dataset
        
    except Exception as e:
        print(f"Error creating balanced generator: {e}")
        print("Returning original generator")
        traceback.print_exc()  # Print full traceback for debugging
        return generator

def fine_tune_model(model, train_generator, val_generator, epochs=5, early_stopping_patience=3, 
                   multi_label=True, class_weights=None, batch_size=32, model_save_path=None):
    """
    Fine-tune the model by unfreezing more layers and training with a smaller learning rate.
    
    Args:
        model: The pre-trained model to fine-tune
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs for fine-tuning
        early_stopping_patience: Patience for early stopping
        multi_label: Whether the model is for multi-label classification
        class_weights: Dictionary of class weights for imbalanced data
        batch_size: Batch size for training
        model_save_path: Path to save the best model during fine-tuning
    
    Returns:
        History object containing training metrics and the fine-tuned model
    """
    try:
        # Clear any existing session
        tf.keras.backend.clear_session()
        
        # Set up mixed precision training
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Get the current strategy
        current_strategy = tf.distribute.get_strategy()
        
        # Create a new model with the same architecture but under the current strategy
        with current_strategy.scope():
            # Create a new model with the same architecture
            new_model = tf.keras.models.clone_model(model)
            new_model.set_weights(model.get_weights())
            
            # Unfreeze more layers for fine-tuning
            for layer in new_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True  # Keep BatchNormalization layers trainable
                elif 'efficientnet' in layer.name:
                    # Unfreeze the last two blocks of EfficientNet
                    if any(block in layer.name for block in ['block6', 'block7']):
                        layer.trainable = True
                    else:
                        layer.trainable = False
                else:
                    layer.trainable = True
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=early_stopping_patience // 2,
                    min_lr=1e-7
                )
            ]
            
            # Add model checkpoint if save path is provided
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                # Ensure the checkpoint path uses .keras extension
                checkpoint_path = os.path.join(os.path.dirname(model_save_path), 'best_model.keras')
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=False
                    )
                )
            
            # Define loss function based on multi-label setting
            if multi_label:
                loss_fn = tf.keras.losses.BinaryCrossentropy()
            else:
                loss_fn = tf.keras.losses.CategoricalCrossentropy()
            
            # Compile model with smaller learning rate
            new_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=loss_fn,
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()
                ]
            )
            
            # Train the model
            history = new_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                batch_size=batch_size
            )
            
            return history, new_model
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        traceback.print_exc()
        return None, None

def setup_gpu(memory_limit=None, allow_growth=True):
    """
    Configure GPU settings to optimize memory usage
    
    Args:
        memory_limit: Memory limit in MB (e.g., 20480 for 20GB)
        allow_growth: Whether to allow GPU memory growth
    
    Returns:
        The appropriate TensorFlow distribution strategy
    """
    global strategy
    
    # Clear any existing session to avoid strategy stack issues
    tf.keras.backend.clear_session()
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPU found. Using CPU strategy.")
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        return strategy
    
    try:
        # Configure GPU memory settings
        for gpu in gpus:
            if allow_growth:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for {gpu}")
                except RuntimeError as e:
                    print(f"Error setting memory growth: {e}")
            
            if memory_limit:
                # Set memory limit in bytes
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit * 1024 * 1024)]
                    )
                    print(f"Memory limit set to {memory_limit} MB for {gpu}")
                except RuntimeError as e:
                    print(f"Error setting memory limit: {e}")
            else:
                # Default to using 16GB for a 24GB GPU to leave headroom
                default_limit = 16384  # 16GB in MB
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=default_limit * 1024 * 1024)]
                    )
                    print(f"Default memory limit set to {default_limit} MB for {gpu}")
                except RuntimeError as e:
                    print(f"Error setting default memory limit: {e}")
        
        # Create a strategy for the first GPU and set globally
        print("Creating GPU strategy...")
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using GPU strategy with device: {strategy.extended.worker_devices[0]}")
        
        # Enable mixed precision for better performance
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled (float16)")
        except Exception as e:
            print(f"Error setting mixed precision: {e}")
        
        # Force a simple operation to ensure the strategy is valid
        with strategy.scope():
            test_tensor = tf.zeros((1, 1))
            _ = test_tensor + 1
            print("Strategy verified with test operation")
        
        return strategy
    
    except Exception as e:
        print(f"Error setting up GPU strategy: {e}")
        print("Falling back to CPU")
        # Ensure we have a valid strategy even if GPU setup fails
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        return strategy

def log_model_to_mlflow(model, history, model_name, fold_idx, class_indices):
    """
    Log model and metrics to MLflow with proper signature and input example
    """
    try:
        # Set experiment
        mlflow.set_experiment("skin-lesion-classification")
        
        # Start a new run
        with mlflow.start_run(run_name=f"{model_name}_fold_{fold_idx}"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("fold_idx", fold_idx)
            mlflow.log_param("num_classes", len(class_indices))
            mlflow.log_param("class_mapping", class_indices)
            
            # Log metrics from training history
            metrics_to_log = {
                'accuracy': 'accuracy',
                'val_accuracy': 'val_accuracy',
                'loss': 'loss',
                'val_loss': 'val_loss',
                'auc': 'auc',
                'val_auc': 'val_auc',
                'precision': 'precision',
                'val_precision': 'val_precision',
                'recall': 'recall',
                'val_recall': 'val_recall'
            }
            
            # Log final and best metrics
            for metric_name, history_key in metrics_to_log.items():
                if history_key in history.history:
                    # Log the final value
                    final_value = history.history[history_key][-1]
                    mlflow.log_metric(f"final_{metric_name}", final_value)
                    
                    # Log the best value
                    if 'val_' in history_key:
                        if 'loss' in history_key:
                            best_value = min(history.history[history_key])
                            best_epoch = history.history[history_key].index(best_value) + 1
                            mlflow.log_metric(f"best_{metric_name}", best_value)
                            mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)
                        else:
                            best_value = max(history.history[history_key])
                            best_epoch = history.history[history_key].index(best_value) + 1
                            mlflow.log_metric(f"best_{metric_name}", best_value)
                            mlflow.log_metric(f"best_{metric_name}_epoch", best_epoch)
            
            # Calculate and log ICBHI score if recall and specificity are available
            if 'val_recall' in history.history and 'val_specificity' in history.history:
                val_icbhi = [(r + s) / 2 for r, s in zip(history.history['val_recall'], history.history['val_specificity'])]
                final_val_icbhi = val_icbhi[-1]
                best_val_icbhi = max(val_icbhi)
                best_val_icbhi_epoch = val_icbhi.index(best_val_icbhi) + 1
                
                mlflow.log_metric("final_val_icbhi_score", final_val_icbhi)
                mlflow.log_metric("best_val_icbhi_score", best_val_icbhi)
                mlflow.log_metric("best_val_icbhi_score_epoch", best_val_icbhi_epoch)
                
            # Log the training curves
            try:
                # Create a figure for the metrics
                plt.figure(figsize=(12, 8))
                
                # Plot each available metric
                metrics_to_plot = [
                    ('loss', 'Loss'),
                    ('accuracy', 'Accuracy'),
                    ('precision', 'Precision'),
                    ('recall', 'Recall/Sensitivity'),
                    ('auc', 'AUC')
                ]
                
                # Create subplots for each metric
                fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4*len(metrics_to_plot)))
                
                for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
                    # Check if metrics exist in history
                    if metric_key in history.history:
                        axs[i].plot(history.history[metric_key], label=f'Training {metric_label}')
                        
                        # Plot validation metrics if available
                        val_key = f'val_{metric_key}'
                        if val_key in history.history:
                            axs[i].plot(history.history[val_key], label=f'Validation {metric_label}')
                        
                        axs[i].set_title(f'{metric_label} During Training')
                        axs[i].set_xlabel('Epoch')
                        axs[i].set_ylabel(metric_label)
                        axs[i].legend()
                        axs[i].grid(True)
                
                # Save the figure to a temporary file
                metrics_plot_path = f"models/metrics_plot_{fold_idx}.png"
                plt.tight_layout()
                plt.savefig(metrics_plot_path)
                plt.close()
                
                # Log the figure to MLflow
                mlflow.log_artifact(metrics_plot_path)
                
                # Clean up
                try:
                    os.remove(metrics_plot_path)
                except:
                    pass
                
            except Exception as plot_error:
                print(f"Error creating and logging plots: {plot_error}")
            
            # Log the model with signature and input example
            try:
                # Save model to a temporary directory with proper .keras extension
                model_path = f"models/temp_model_{fold_idx}.keras"
                model.save(model_path)
                
                # Create an input example (dummy input with correct shape)
                input_example = np.zeros((1, 224, 224, 3), dtype=np.float32)
                
                # Create model signature
                signature = infer_signature(input_example, model.predict(input_example))
                
                # Log the model to MLflow with signature and input example
                mlflow.tensorflow.log_model(
                    model, 
                    "model",
                    signature=signature,
                    input_example=input_example
                )
                
                # Clean up
                if os.path.exists(model_path):
                    try:
                     os.remove(model_path)
                    except:
                        pass
            except Exception as e:
                print(f"Error logging model to MLflow: {e}")
    
    except Exception as e:
        print(f"Error in MLflow logging: {e}")
        traceback.print_exc()


def create_ensemble_model(num_classes, num_models=3):
    """
    Create an ensemble of models for better performance on imbalanced data
    """
    models = []
    
    for i in range(num_models):
        with strategy.scope():
            # Create base model
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            
            # Freeze base model
            base_model.trainable = False
            
            # Build model
            inputs = Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile with focal loss
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=focal_loss(gamma=2.0, alpha=0.25),
                metrics=['accuracy', Recall(name='recall')]
            )
            
            models.append(model)
    
    # Create ensemble model
    ensemble_input = Input(shape=(224, 224, 3))
    outputs = [model(ensemble_input) for model in models]
    ensemble_output = tf.keras.layers.Average()(outputs)
    
    ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output)
    
    # Compile ensemble model
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', Recall(name='recall')]
    )
    
    return ensemble_model, models