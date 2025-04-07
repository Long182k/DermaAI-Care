from mlflow.models.signature import infer_signature
from PIL import Image
from sklearn.model_selection import KFold
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
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
def build_peft_model(num_classes, r=8, alpha=32, multi_label=False):
    """
    Build and compile the model with PEFT (LoRA) optimizations and memory efficiency
    
    Args:
        num_classes: Number of output classes
        r: Rank of the low-rank adaptation matrices
        alpha: Scaling factor for the adaptation
        multi_label: Whether to use multi-label classification (sigmoid activation)
    """
    print(f"Building model with {num_classes} classes, multi_label={multi_label}")
    
    # Clear any existing session to ensure consistent strategy
    tf.keras.backend.clear_session()
    
    # Enable mixed precision for faster training and lower memory usage
    original_policy = tf.keras.mixed_precision.global_policy()
    set_global_policy('mixed_float16')
    
    try:
        # Use the global strategy defined at the module level
        with strategy.scope():
            # Create the base model with efficient memory usage
            base_model = InceptionResNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            # Build the model using Functional API
            inputs = Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            
            # Apply LoRA to the final dense layer
            dense_layer = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense_1')
            dense_output = dense_layer(x)
            
            # Add LoRA adaptation
            lora_adaptation = LoRALayer(512, 512, r=r, alpha=alpha, name='lora_adaptation')(dense_output)
            
            # Combine original dense output with LoRA adaptation
            combined = tf.keras.layers.Add()([dense_output, lora_adaptation])
            
            x = Dropout(0.5)(combined)
            
            # Use sigmoid activation for multi-label, softmax for single-label
            final_activation = 'sigmoid' if multi_label else 'softmax'
            outputs = Dense(num_classes, activation=final_activation, kernel_initializer='he_normal')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Choose appropriate loss and metrics based on the task
            if multi_label:
                loss = tf.keras.losses.BinaryCrossentropy()
                metrics = [
                    BinaryAccuracy(name='accuracy'),
                    AUC(name='auc', multi_label=True),
                    Precision(name='precision'),
                    Recall(name='recall')
                ]
            else:
                loss = focal_loss(gamma=2.0, alpha=0.25)
                metrics = [
                    'accuracy',
                    AUC(name='auc', from_logits=False),
                    Precision(name='precision'),
                    Recall(name='recall')
                ]
            
            # Compile the model with memory-efficient settings and proper metrics
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=loss,
                metrics=metrics
            )
            
            return model
            
    except Exception as e:
        print(f"Error building model: {e}")
        # Restore original policy on error
        tf.keras.mixed_precision.set_global_policy(original_policy)
        # Try with a simpler model on error
        return None

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

# Analysis of Training and Evaluation Results

# Based on the logs you've provided, your model is running without crashing, which is good news after the previous errors. However, there are some performance issues that need attention:

## Current Status

# 1. **Sample Mismatch Fixed**: The code successfully handled the mismatch between true labels (19) and predictions (16) by adjusting to 16 samples.

# 2. **Class Imbalance**: Your dataset is imbalanced with 12 samples of class 0 and 4 samples of class 1.

# 3. **Performance Issues**:
#    - The model is predicting all samples as class 0 (majority class)
#    - Precision, recall, and F1-score for class 1 are all 0.0
#    - Overall accuracy is 75%, but this is misleading due to class imbalance

# 4. **Warnings**: The sklearn warnings about "Precision and F-score are ill-defined" indicate that the model didn't predict any samples for class 1.

## Recommendations

# To improve your model's performance, I suggest the following modifications to your training approach:

# In the train_model function, modify the class weights handling:

# Modify the train_model function to better handle class imbalance

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
    Train the model with proper callbacks and monitoring
    
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
        batch_size: Batch size for training, will be reduced if OOM occurs
    """
    print(f"Using batch size: {batch_size}")
    
    # Set up callbacks
    monitor_metric = 'val_loss' if multi_label else 'val_accuracy'
    mode = 'min' if monitor_metric == 'val_loss' else 'max'
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode=mode
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/checkpoint.keras',
            monitor=monitor_metric,
            save_best_only=True,
            mode=mode,
            verbose=1
        ),
        TimeoutCallback(timeout_seconds=3600),  # 1 hour timeout
        MemoryCleanupCallback(cleanup_frequency=2)  # More frequent cleanup
    ]
    
    # Store the original strategy from the model
    model_strategy = None
    for layer in model.layers:
        if hasattr(layer, '_strategy'):
            model_strategy = layer._strategy
            break
    
    # For multi-label classification with class weights, we need to handle it differently
    if multi_label and class_weights:
        print(f"Applying class weights for multi-label classification: {class_weights}")
        
        # Important: Clear the session to avoid strategy mixing
        tf.keras.backend.clear_session()
        
        # Clone the model to ensure we're using a fresh model with consistent strategy
        orig_weights = model.get_weights()
        config = model.get_config()
        
        # If we detected a strategy, use it to recreate the model
        if model_strategy:
            with model_strategy.scope():
                model = tf.keras.Model.from_config(config)
                model.set_weights(orig_weights)
        else:
            # If no strategy detected, recreate with the global strategy
            with strategy.scope():
                model = tf.keras.Model.from_config(config)
                model.set_weights(orig_weights)
        
        try:
            # Memory optimization
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        # Try to set memory growth, but don't error if already initialized
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        except:
                            print("GPU memory growth already set")
                except Exception as e:
                    print(f"Error setting memory growth: {e}")
            
            # Temporarily disable mixed precision for better stability with custom loss
            original_policy = tf.keras.mixed_precision.global_policy()
            tf.keras.mixed_precision.set_global_policy('float32')
            
            # Create custom weighted binary crossentropy with float32 precision
            def weighted_binary_crossentropy(y_true, y_pred):
                # Cast inputs to float32 for numerical stability
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                
                # Standard binary crossentropy calculation
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
                
                # Create per-class weights tensor
                weights_tensor = tf.convert_to_tensor(
                    [float(class_weights.get(i, 1.0)) for i in range(model.output_shape[-1])],
                    dtype=tf.float32
                )
                
                # Apply class-specific weights to positive examples only
                class_weighted_bce = bce * (y_true * weights_tensor + (1 - y_true))
                
                # Average over class axis first, then batch
                return tf.reduce_mean(tf.reduce_mean(class_weighted_bce, axis=-1))
            
            # Use the same strategy for compilation
            if model_strategy:
                with model_strategy.scope():
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=weighted_binary_crossentropy,
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.AUC(name='auc', multi_label=True),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
            else:
                with strategy.scope():
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=weighted_binary_crossentropy,
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.AUC(name='auc', multi_label=True),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
            
            # Use the provided batch size (will be reduced if OOM occurs)
            try_batch_size = batch_size
            
            # Train the model with the provided batch size
            print(f"Training with batch size of {try_batch_size}")
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks,
                batch_size=try_batch_size,
                verbose=1,
                class_weight=None  # We're handling weights in the loss function
            )
            
            # Restore original mixed precision policy
            tf.keras.mixed_precision.set_global_policy(original_policy)
            
            # Run garbage collection to free memory
            gc.collect()
            
            return history
            
        except Exception as e:
            print(f"Error during training with custom loss: {e}")
            traceback.print_exc()
            
            # Cleanup and memory management
            gc.collect()
            if 'original_policy' in locals():
                tf.keras.mixed_precision.set_global_policy(original_policy)
            
            # Fallback: Try with an even smaller batch size
            try:
                print("Attempting training with a smaller batch size...")
                # Use an even smaller batch size
                try_batch_size = batch_size // 2
                
                # Reset the TensorFlow session
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Recreate model with consistent strategy
                if model_strategy:
                    with model_strategy.scope():
                        model = tf.keras.Model.from_config(config)
                        model.set_weights(orig_weights)
                        
                        # Standard binary crossentropy
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=[
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.AUC(name='auc', multi_label=True),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ]
                        )
                else:
                    with strategy.scope():
                        model = tf.keras.Model.from_config(config)
                        model.set_weights(orig_weights)
                        
                        # Standard binary crossentropy
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=[
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.AUC(name='auc', multi_label=True),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ]
                        )
                
                # Try with the smaller batch size
                print(f"Training with reduced batch size of {try_batch_size}")
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=epochs,
                    callbacks=callbacks,
                    batch_size=try_batch_size,
                    verbose=1,
                    class_weight=None
                )
                print("Training completed with reduced batch size")
                return history
                
            except Exception as fallback_error:
                print(f"Fallback training also failed: {fallback_error}")
                print("Falling back to minimal training")
                
                # Last resort: Try with minimal settings
                try:
                    # One more attempt with even smaller batch size
                    try_batch_size = max(8, batch_size // 4)
                    
                    # Reset the TensorFlow session
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # Recreate model with consistent strategy
                    if model_strategy:
                        with model_strategy.scope():
                            model = tf.keras.Model.from_config(config)
                            model.set_weights(orig_weights)
                            
                            model.compile(
                                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate/10),
                                loss='binary_crossentropy',
                                metrics=['accuracy']
                            )
                    else:
                        with strategy.scope():
                            model = tf.keras.Model.from_config(config)
                            model.set_weights(orig_weights)
                            
                            model.compile(
                                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate/10),
                                loss='binary_crossentropy',
                                metrics=['accuracy']
                            )
                    
                    history = model.fit(
                        train_data,
                        validation_data=val_data,
                        epochs=min(5, epochs),  # Fewer epochs
                        callbacks=[TimeoutCallback(timeout_seconds=1800)],
                        batch_size=try_batch_size,  # Very small batch size
                        verbose=1,
                        class_weight=None
                    )
                    return history
                except Exception as minimal_error:
                    print(f"Minimal training also failed: {minimal_error}")
                    return None
    
    # For standard training without class weights or for single-label
    else:
        # Train the model with standard parameters but use the provided batch size
        try:
            # Reset the TensorFlow session
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Recreate model with consistent strategy
            orig_weights = model.get_weights()
            config = model.get_config()
            
            if model_strategy:
                with model_strategy.scope():
                    model = tf.keras.Model.from_config(config)
                    model.set_weights(orig_weights)
                    
                    # Re-compile with the same parameters
                    if multi_label:
                        loss = tf.keras.losses.BinaryCrossentropy()
                    else:
                        loss = focal_loss(gamma=2.0, alpha=0.25)
                        
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=loss,
                        metrics=model.metrics
                    )
            else:
                with strategy.scope():
                    model = tf.keras.Model.from_config(config)
                    model.set_weights(orig_weights)
                    
                    # Re-compile with the same parameters
                    if multi_label:
                        loss = tf.keras.losses.BinaryCrossentropy()
                    else:
                        loss = focal_loss(gamma=2.0, alpha=0.25)
                        
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=loss,
                        metrics=model.metrics
                    )
            
            try_batch_size = batch_size
            print(f"Training with batch size of {try_batch_size}")
            
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks,
                batch_size=try_batch_size,
                verbose=1,
                class_weight=class_weights if not multi_label else None
            )
            return history
            
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            
            try:
                print("Attempting training with a smaller batch size...")
                try_batch_size = batch_size // 2
                
                # Reset the TensorFlow session
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Recreate model with consistent strategy
                if model_strategy:
                    with model_strategy.scope():
                        model = tf.keras.Model.from_config(config)
                        model.set_weights(orig_weights)
                        
                        # Re-compile with the same parameters
                        if multi_label:
                            loss = tf.keras.losses.BinaryCrossentropy()
                        else:
                            loss = focal_loss(gamma=2.0, alpha=0.25)
                            
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=loss,
                            metrics=model.metrics
                        )
                else:
                    with strategy.scope():
                        model = tf.keras.Model.from_config(config)
                        model.set_weights(orig_weights)
                        
                        # Re-compile with the same parameters
                        if multi_label:
                            loss = tf.keras.losses.BinaryCrossentropy()
                        else:
                            loss = focal_loss(gamma=2.0, alpha=0.25)
                            
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=loss,
                            metrics=model.metrics
                        )
                
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=epochs,
                    callbacks=callbacks,
                    batch_size=try_batch_size,
                    verbose=1,
                    class_weight=class_weights if not multi_label else None
                )
                return history
                
            except Exception as fallback_error:
                print(f"Fallback training also failed: {fallback_error}")
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

def fine_tune_model(
    model, 
    train_generator, 
    val_generator, 
    epochs, 
    early_stopping_patience,
    multi_label=False,
    class_weights=None,
    batch_size=24
):
    """
    Fine-tune the model by unfreezing some layers and training with a lower learning rate
    
    Args:
        model: The pre-trained model to fine-tune
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs for fine-tuning
        early_stopping_patience: Number of epochs to wait before early stopping
        multi_label: Whether this is a multi-label classification task
        class_weights: Dictionary of class weights for imbalanced datasets
        batch_size: Batch size for training, will be reduced if OOM occurs
    """
    print(f"Starting fine-tuning process with batch size: {batch_size}")
    
    # Clear session to avoid strategy mixing issues
    tf.keras.backend.clear_session()
    
    # Store the original strategy from the model
    model_strategy = None
    for layer in model.layers:
        if hasattr(layer, '_strategy'):
            model_strategy = layer._strategy
            break
    
    # Create a clone of the model to ensure consistent strategy
    model_config = model.get_config()
    weights = model.get_weights()
    
    # Define callbacks outside the try blocks so they're accessible everywhere
    monitor_metric = 'val_loss' if multi_label else 'val_accuracy'
    mode = 'min' if monitor_metric == 'val_loss' else 'max'
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode=mode
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/fine_tuned_checkpoint.keras',
            monitor=monitor_metric,
            save_best_only=True,
            mode=mode,
            verbose=1
        ),
        TimeoutCallback(timeout_seconds=3600),
        MemoryCleanupCallback(cleanup_frequency=2)  # More frequent cleanup
    ]
    
    # Recreate the model with the appropriate strategy
    if model_strategy:
        with model_strategy.scope():
            new_model = tf.keras.Model.from_config(model_config)
            new_model.set_weights(weights)
            
            # Unfreeze the last few layers of the base model
            base_model = new_model.layers[1]  # Get the InceptionResNetV2 model
            for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
                layer.trainable = True
    else:
        # If no strategy detected, recreate with the global strategy
        with strategy.scope():
            new_model = tf.keras.Model.from_config(model_config)
            new_model.set_weights(weights)
            
            # Unfreeze the last few layers of the base model
            base_model = new_model.layers[1]  # Get the InceptionResNetV2 model
            for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
                layer.trainable = True
    
    # For multi-label classification with class weights, we need a custom loss
    if multi_label and class_weights:
        print(f"Applying class weights for fine-tuning: {class_weights}")
        
        try:
            # Memory optimization step: Add memory safety layer
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        # Try to set memory growth, but don't error if already initialized
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        except:
                            print("GPU memory growth already set")
                except Exception as e:
                    print(f"Error setting memory growth: {e}")
            
            # Temporarily disable mixed precision for better stability with custom loss
            original_policy = tf.keras.mixed_precision.global_policy()
            tf.keras.mixed_precision.set_global_policy('float32')
            
            # Create custom weighted binary crossentropy with float32 precision
            def weighted_binary_crossentropy(y_true, y_pred):
                # Cast inputs to float32 for numerical stability
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                
                # Standard binary crossentropy calculation
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
                
                # Create per-class weights tensor
                weights_tensor = tf.convert_to_tensor(
                    [float(class_weights.get(i, 1.0)) for i in range(new_model.output_shape[-1])],
                    dtype=tf.float32
                )
                
                # Apply class-specific weights to positive examples only
                class_weighted_bce = bce * (y_true * weights_tensor + (1 - y_true))
                
                # Average over class axis first, then batch
                return tf.reduce_mean(tf.reduce_mean(class_weighted_bce, axis=-1))
            
            # Compile with our custom loss using the same strategy
            if model_strategy:
                with model_strategy.scope():
                    new_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                        loss=weighted_binary_crossentropy,
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.AUC(name='auc', multi_label=True),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
            else:
                with strategy.scope():
                    new_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                        loss=weighted_binary_crossentropy,
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.AUC(name='auc', multi_label=True),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
                        ]
                    )
            
            # Use provided batch size
            try_batch_size = batch_size
            
            # Train the model with standard fit API but no class_weight parameter
            print(f"Fine-tuning with batch size of {try_batch_size}")
            history = new_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,  # Use the predefined callbacks
                batch_size=try_batch_size,
                verbose=1,
                class_weight=None  # Don't use class_weight as we handle it in the loss
            )
            
            # Copy weights back to original model
            model.set_weights(new_model.get_weights())
            
            # Clean up memory
            del new_model
            gc.collect()
            
            # Restore original mixed precision policy
            tf.keras.mixed_precision.set_global_policy(original_policy)
            
            return history
            
        except Exception as e:
            print(f"Error during fine-tuning with custom loss: {e}")
            traceback.print_exc()
            
            # Memory cleanup
            gc.collect()
            if 'original_policy' in locals():
                tf.keras.mixed_precision.set_global_policy(original_policy)
            
            # Fallback with smaller batch size
            try:
                print("Trying with a smaller batch size...")
                # Reset the model and session
                tf.keras.backend.clear_session()
                gc.collect()
                
                try_batch_size = batch_size // 2
                
                # Recreate model with consistent strategy
                if model_strategy:
                    with model_strategy.scope():
                        new_model = tf.keras.Model.from_config(model_config)
                        new_model.set_weights(weights)
                        
                        # Only unfreeze last 15 layers to save memory
                        base_model = new_model.layers[1]
                        base_model.trainable = False
                        for layer in base_model.layers[-15:]:
                            layer.trainable = True
                        
                        # Compile with standard loss
                        new_model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),  # Lower learning rate
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=[
                                BinaryAccuracy(name='accuracy'),
                                AUC(name='auc', multi_label=True),
                                Precision(name='precision'),
                                Recall(name='recall')
                            ]
                        )
                else:
                    with strategy.scope():
                        new_model = tf.keras.Model.from_config(model_config)
                        new_model.set_weights(weights)
                        
                        # Only unfreeze last 15 layers to save memory
                        base_model = new_model.layers[1]
                        base_model.trainable = False
                        for layer in base_model.layers[-15:]:
                            layer.trainable = True
                        
                        # Compile with standard loss
                        new_model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),  # Lower learning rate
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=[
                                BinaryAccuracy(name='accuracy'),
                                AUC(name='auc', multi_label=True),
                                Precision(name='precision'),
                                Recall(name='recall')
                            ]
                        )
                
                # Try with smaller batch size
                print(f"Fine-tuning with reduced batch size of {try_batch_size}")
                history = new_model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    callbacks=callbacks,  # Use the predefined callbacks
                    batch_size=try_batch_size,
                    class_weight=None,
                    verbose=1
                )
                
                # Copy weights back to original model
                model.set_weights(new_model.get_weights())
                
                # Clean up memory
                del new_model
                gc.collect()
                
                return history
                
            except Exception as inner_e:
                print(f"Error during fallback fine-tuning: {inner_e}")
                
                # Last resort with minimal setup
                try:
                    print("Trying minimal fine-tuning setup...")
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    try_batch_size = max(8, batch_size // 4)
                    
                    # Recreate with minimal trainable parameters
                    if model_strategy:
                        with model_strategy.scope():
                            new_model = tf.keras.Model.from_config(model_config)
                            new_model.set_weights(weights)
                            
                            # Only unfreeze classification layers
                            for layer in new_model.layers:
                                layer.trainable = False
                            new_model.layers[-1].trainable = True  # Only final Dense layer
                            
                            # Simplified compilation
                            new_model.compile(
                                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                                loss='binary_crossentropy',
                                metrics=['accuracy']
                            )
                    else:
                        with strategy.scope():
                            new_model = tf.keras.Model.from_config(model_config)
                            new_model.set_weights(weights)
                            
                            # Only unfreeze classification layers
                            for layer in new_model.layers:
                                layer.trainable = False
                            new_model.layers[-1].trainable = True  # Only final Dense layer
                            
                            # Simplified compilation
                            new_model.compile(
                                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                                loss='binary_crossentropy',
                                metrics=['accuracy']
                            )
                    
                    # Use minimal callbacks for last resort
                    minimal_callbacks = [
                        TimeoutCallback(timeout_seconds=1800)
                    ]
                    
                    # Minimal training
                    history = new_model.fit(
                        train_generator,
                        validation_data=val_generator,
                        epochs=min(3, epochs),  # Reduced epochs
                        callbacks=minimal_callbacks,  # Use minimal callbacks
                        batch_size=try_batch_size,  # Very small batch
                        verbose=1
                    )
                    
                    model.set_weights(new_model.get_weights())
                    del new_model
                    gc.collect()
                    return history
                    
                except Exception as final_e:
                    print(f"All fine-tuning attempts failed: {final_e}")
                    traceback.print_exc()
                    return None
            
    else:
        # For single-label or non-weighted scenarios, use standard approach with smaller batch
        if multi_label:
            loss = tf.keras.losses.BinaryCrossentropy()
            print("Using binary crossentropy for multi-label fine-tuning")
        else:
            loss = focal_loss(gamma=2.0, alpha=0.25)
            print("Using focal loss for single-label fine-tuning")
        
        # Set up appropriate metrics based on task type
        if multi_label:
            metrics = [
                BinaryAccuracy(name='accuracy'),
                AUC(name='auc', multi_label=True),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        else:
            metrics = [
                'accuracy',
                tf.keras.metrics.AUC(name='auc', from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        # Compile the model with appropriate loss and metrics using the same strategy
        if model_strategy:
            with model_strategy.scope():
                new_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
                    loss=loss,
                    metrics=metrics
                )
        else:
            with strategy.scope():
                new_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
                    loss=loss,
                    metrics=metrics
                )
        
        # Fine-tune the model with provided batch size
        try:
            try_batch_size = batch_size
            print(f"Fine-tuning with batch size of {try_batch_size}")
            
            history = new_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,  # Use the predefined callbacks
                batch_size=try_batch_size,
                class_weight=class_weights if not multi_label else None,
                verbose=1
            )
            
            # Copy weights back to original model to maintain the reference
            model.set_weights(new_model.get_weights())
            
            # Clear up memory
            del new_model
            gc.collect()
            
            return history
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            traceback.print_exc()
            
            # Try with smaller batch size
            try:
                print("Trying with a smaller batch size...")
                tf.keras.backend.clear_session()
                gc.collect()
                
                try_batch_size = batch_size // 2
                
                # Recreate model with consistent strategy
                if model_strategy:
                    with model_strategy.scope():
                        new_model = tf.keras.Model.from_config(model_config)
                        new_model.set_weights(weights)
                        
                        # Unfreeze fewer layers to save memory
                        base_model = new_model.layers[1]
                        base_model.trainable = False
                        for layer in base_model.layers[-15:]:  # Only 15 layers instead of 30
                            layer.trainable = True
                        
                        # Recompile
                        new_model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                            loss=loss,
                            metrics=metrics
                        )
                else:
                    with strategy.scope():
                        new_model = tf.keras.Model.from_config(model_config)
                        new_model.set_weights(weights)
                        
                        # Unfreeze fewer layers to save memory
                        base_model = new_model.layers[1]
                        base_model.trainable = False
                        for layer in base_model.layers[-15:]:  # Only 15 layers instead of 30
                            layer.trainable = True
                        
                        # Recompile
                        new_model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                            loss=loss,
                            metrics=metrics
                        )
                
                # Try with smaller batch
                print(f"Fine-tuning with reduced batch size of {try_batch_size}")
                history = new_model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    callbacks=callbacks,  # Use the predefined callbacks
                    batch_size=try_batch_size,
                    class_weight=class_weights if not multi_label else None,
                    verbose=1
                )
                
                model.set_weights(new_model.get_weights())
                del new_model
                gc.collect()
                return history
                
            except Exception as final_e:
                print(f"Fine-tuning failed with all attempts: {final_e}")
                traceback.print_exc()
                return None

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
            base_model = InceptionResNetV2(
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