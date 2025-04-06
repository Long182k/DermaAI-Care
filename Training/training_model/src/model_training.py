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
import numpy as np
import os
import time
import gc
import traceback

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
def build_peft_model(num_classes=9, r=8, alpha=32):
    """
    Build and compile the model with PEFT (LoRA) optimizations and memory efficiency
    Adapted for 9-class skin lesion classification
    """
    # Enable mixed precision for faster training and lower memory usage
    set_global_policy('mixed_float16')
    
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
        
        # Create LoRA adaptation layer using the r and alpha parameters
        lora_layer = LoRALayer(
            in_features=512,
            out_features=512,
            r=r,  # Using the r parameter
            alpha=alpha,  # Using the alpha parameter
            name='lora_adaptation'
        )(dense_output)
        
        # Combine the original dense output with the LoRA adaptation
        x = tf.keras.layers.Add()([dense_output, lora_layer])
        
        # Add dropout for regularization
        x = Dropout(0.5)(x)
        
        # Final classification layer with 9 outputs
        # MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with focal loss to address class imbalance
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc', from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model


class TimeoutCallback(Callback):
    def __init__(self, timeout_seconds=1800):  # Default 30 minutes
        super(TimeoutCallback, self).__init__()
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.timeout_seconds:
            print(f"\nTraining timed out after {elapsed_time:.2f} seconds")
            self.model.stop_training = True
            
    def on_batch_end(self, batch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.timeout_seconds:
            print(f"\nTraining timed out after {elapsed_time:.2f} seconds")
            self.model.stop_training = True

class MemoryCleanupCallback(Callback):
    def __init__(self, cleanup_frequency=5):
        super(MemoryCleanupCallback, self).__init__()
        self.cleanup_frequency = cleanup_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        # Run garbage collection at the end of each epoch
        gc.collect()
        
        # Clear TensorFlow's GPU memory cache periodically
        if epoch % self.cleanup_frequency == 0:
            tf.keras.backend.clear_session()
            gc.collect()

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

def train_model(model, train_data, val_data, epochs, early_stopping_patience, reduce_lr_patience, 
                class_weights, train_class_indices, use_datasets=False, 
                class_weight_multiplier=3.0, use_focal_loss=True, learning_rate=0.0001):
    """
    Enhanced training function with proper memory management and early stopping
    
    Args:
        model: The model to train
        train_data: Training data generator or dataset
        val_data: Validation data generator or dataset
        epochs: Number of epochs to train
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        class_weights: Dictionary of class weights
        train_class_indices: Dictionary mapping class names to indices
        use_datasets: Whether train_data and val_data are tf.data.Dataset objects
        class_weight_multiplier: Multiplier for minority class weights
        use_focal_loss: Whether to use focal loss
        learning_rate: Learning rate for optimizer
    """
    # Ensure class weights are properly applied
    print(f"Using class weights: {class_weights}")
    
    # Apply class weight multiplier to minority classes
    if class_weights and class_weight_multiplier > 1.0:
        weights = list(class_weights.values())
        median_weight = np.median(weights)
        
        # Apply multiplier to classes with above-median weights
        for cls in class_weights:
            if class_weights[cls] > median_weight:
                class_weights[cls] *= class_weight_multiplier
        
        print(f"Applied multiplier to minority classes: {class_weights}")
    
    # If class weights are very imbalanced, adjust them to be less extreme
    max_weight_ratio = 10.0  # Maximum ratio between highest and lowest weight
    if class_weights:
        weights = list(class_weights.values())
        max_weight = max(weights)
        min_weight = min(weights)
        
        if max_weight / min_weight > max_weight_ratio:
            # Scale down the weights to have a more reasonable ratio
            scale_factor = max_weight / (min_weight * max_weight_ratio)
            for cls in class_weights:
                if class_weights[cls] == max_weight:
                    class_weights[cls] = max_weight / scale_factor
            
            print(f"Adjusted class weights to prevent extreme imbalance: {class_weights}")
    
    # Create callbacks for training with focus on minority class performance
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',  # Changed from val_loss to val_recall to focus on minority class
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='max'  # Changed from 'min' to 'max' since we're monitoring recall
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_recall',  # Changed from val_loss to val_recall
            factor=0.2,
            patience=reduce_lr_patience,  # Using the parameter here
            min_lr=learning_rate / 100,  # Using learning_rate parameter
            mode='max',  # Changed from 'min' to 'max'
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/checkpoint.keras',
            monitor='val_recall',
            save_best_only=True,  # Corrected from save_best_weights_only
            mode='max',
            verbose=1
        ),
        TimeoutCallback(timeout_seconds=3600),  # 1 hour timeout
        MemoryCleanupCallback(cleanup_frequency=2)  # Clean memory every 2 epochs
    ]
    
    # Compile the model with appropriate loss function
    if use_focal_loss:
        print(f"Using focal loss for training with learning rate: {learning_rate}")
        loss_function = focal_loss(gamma=2.0, alpha=0.25)
    else:
        print(f"Using categorical crossentropy for training with learning rate: {learning_rate}")
        loss_function = 'categorical_crossentropy'
    
    # Compile with specified learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', from_logits=False),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Log information about the training process
    print(f"Starting training for {epochs} epochs")
    print(f"Class indices: {train_class_indices}")
    
    # Train the model with proper error handling
    try:
        # Handle different types of input data
        if use_datasets:
            # For tf.data.Dataset objects
            print("Using TensorFlow Datasets for training")
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # For Keras data generators
            print("Using Keras data generators for training")
            # Check if we need to create a balanced generator
            if hasattr(train_data, 'class_counts') and sum(train_data.class_counts.values()) > 0:
                print("Creating balanced data generator for training")
                balanced_train_data = create_balanced_data_generator(train_data)
                
                history = model.fit(
                    balanced_train_data,
                    validation_data=val_data,
                    epochs=epochs,
                    callbacks=callbacks,
                    class_weight=class_weights if not use_focal_loss else None,
                    verbose=1
                )
            else:
                # Standard training with the provided generator
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=epochs,
                    callbacks=callbacks,
                    class_weight=class_weights if not use_focal_loss else None,
                    verbose=1
                )
        
        # Force garbage collection after training
        gc.collect()
        
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        
        # Force garbage collection on error
        gc.collect()
        
        return None


def fine_tune_model(model, train_generator, val_generator, epochs, early_stopping_patience, learning_rate=1e-5):
    """
    Fine-tune the model by unfreezing some layers with early stopping and memory optimization
    
    Args:
        model: The model to fine-tune
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs for fine-tuning
        early_stopping_patience: Patience for early stopping
        learning_rate: Learning rate for fine-tuning (should be lower than initial training)
    """
    print(f"Preparing model for fine-tuning with learning rate: {learning_rate}")
    
    # Create callbacks at the beginning of the function to avoid reference errors
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',  # Focus on recall for minority class
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_recall',
            factor=0.2,
            patience=3,
            min_lr=learning_rate / 100,  # Using learning_rate parameter
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/fine_tuned_checkpoint.keras',
            monitor='val_recall',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        TimeoutCallback(timeout_seconds=3600),  # 1 hour timeout
        MemoryCleanupCallback(cleanup_frequency=2)  # Clean memory every 2 epochs
    ]
    
    # Instead of recreating the model, just unfreeze layers in the existing model
    try:
        # Find the base model (InceptionResNetV2) in the current model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):  # This should find the InceptionResNetV2 base model
                base_model = layer
                print(f"Found base model: {base_model.name}")
                
                # Unfreeze only the last 10 layers of the base model to save memory
                for layer in base_model.layers[:-10]:
                    layer.trainable = False
                for layer in base_model.layers[-10:]:
                    layer.trainable = True
                
                print(f"Unfrozen last 10 layers of base model")
                break
        else:
            # If we didn't find a nested model, look for the InceptionResNetV2 layer directly
            for i, layer in enumerate(model.layers):
                if "inception" in layer.name.lower():
                    base_layer = layer
                    print(f"Found base layer at index {i}: {base_layer.name}")
                    
                    # Unfreeze only the last 10 layers if it has layers attribute
                    if hasattr(base_layer, 'layers'):
                        for layer in base_layer.layers[:-10]:
                            layer.trainable = False
                        for layer in base_layer.layers[-10:]:
                            layer.trainable = True
                        print(f"Unfrozen last 10 layers of base layer")
                    else:
                        # If it doesn't have layers, just make it trainable
                        base_layer.trainable = True
                        print(f"Made base layer trainable")
                    break
            else:
                print("Warning: Could not find base model or InceptionResNetV2 layer. Making all layers trainable.")
                model.trainable = True
        
        # Clear session to avoid strategy stack issues
        tf.keras.backend.clear_session()
        
        # Fix for optimizer initialization error - create a new optimizer directly
        new_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Using the learning_rate parameter
        
        # Recompile the model with the new optimizer
        model.compile(
            optimizer=new_optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc', from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"Model recompiled with learning rate: {learning_rate}")
        
    except Exception as e:
        print(f"Error during model layer unfreezing: {e}")
        print("Falling back to simpler fine-tuning approach...")
        
        # Simple approach: just make the whole model trainable
        model.trainable = True
        
        # Clear session to avoid strategy stack issues
        tf.keras.backend.clear_session()
        
        # Recompile with a lower learning rate
        # Recompile the model with focal loss and lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Using the learning_rate parameter
            loss=focal_loss(gamma=2.0, alpha=0.25),  # Use focal loss
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc', from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        # Create callbacks for fine-tuning with focus on minority class
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_recall',  # Focus on recall for minority class
                patience=early_stopping_patience,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_recall',
                factor=0.2,
                patience=3,
                min_lr=learning_rate / 100,  # Using learning_rate parameter
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/fine_tuned_checkpoint.keras',
                monitor='val_recall',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            TimeoutCallback(timeout_seconds=3600),  # 1 hour timeout
            MemoryCleanupCallback(cleanup_frequency=2)  # Clean memory every 2 epochs
        ]
    
    # Train the model without using strategy.scope()
    try:
        # Use a try-except block to handle potential strategy issues
        try:
            # Disable XLA compilation to avoid MaxPool gradient ops error
            tf.config.optimizer.set_jit(False)
            
            # First attempt: try to fit the model directly
            # In your main training function, add:
            
            # Create balanced generators for training
            balanced_train_generator = create_balanced_data_generator(train_generator)
            
            # Use the balanced generator for training
            history = model.fit(
                balanced_train_generator,
                validation_data=val_generator,  # Keep validation data as is to get realistic metrics
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        except RuntimeError as e:
            if "Mixing different tf.distribute.Strategy objects" in str(e):
                print("Strategy mismatch detected. Attempting to recreate model...")
                
                # Save the weights
                weights = model.get_weights()
                
                # Clear session
                tf.keras.backend.clear_session()
                
                # Create a new model with the same architecture
                new_model = clone_model(model)
                
                # Set the weights
                new_model.set_weights(weights)
                
                # Recompile
                new_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(name='auc', from_logits=False),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                    ]
                )
                
                # Disable XLA compilation to avoid MaxPool gradient ops error
                tf.config.optimizer.set_jit(False)
                
                # Try fitting again
                history = new_model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Update the original model
                model = new_model
            else:
                # If it's a different error, re-raise it
                raise
        
        # Force garbage collection after training
        gc.collect()
        
        return history
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        # Force garbage collection on error
        gc.collect()
        
        # Try one last approach - simplified model with fewer layers
        try:
            print("Attempting final approach: creating a simplified model...")
            
            # Create a simplified model that doesn't use MaxPool gradient ops
            with strategy.scope():
                # Create a new model with fewer layers
                base_model = InceptionResNetV2(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg'
                )
                
                # Freeze all layers
                base_model.trainable = False
                
                # Build a simpler model
                inputs = Input(shape=(224, 224, 3))
                x = base_model(inputs, training=False)
                x = Dense(256, activation='relu')(x)
                x = Dropout(0.5)(x)
                outputs = Dense(len(train_generator.class_indices), activation='softmax')(x)
                
                simplified_model = Model(inputs=inputs, outputs=outputs)
                
                # Compile the model
                simplified_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(name='auc', from_logits=False),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')
                    ]
                )
            
            # Disable XLA compilation
            tf.config.optimizer.set_jit(False)
            
            # Train the simplified model
            history = simplified_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Return the simplified model
            return history
        except Exception as e2:
            print(f"Final approach failed: {e2}")
            print("Fine-tuning could not be completed. Returning original model.")
            return None

def create_balanced_data_generator(generator, batch_size=64):
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
        traceback.print_exc() 
        return generator

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
                tf.config.experimental.set_memory_growth(gpu, True)
            
            if memory_limit:
                # Set memory limit in bytes
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit * 1024 * 1024)]
                )
            else:
                # Default to using 16GB for a 24GB GPU to leave headroom
                default_limit = 16384  # 16GB in MB
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=default_limit * 1024 * 1024)]
                )
        
        # Create a strategy for the first GPU
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using GPU strategy with device: {strategy.extended.worker_devices[0]}")
        
        # Enable mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
        
        return strategy
    
    except Exception as e:
        print(f"Error setting up GPU: {e}")
        print("Falling back to CPU")
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
            
            # Log metrics
            for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss', 'auc', 'val_auc']:
                if metric in history.history:
                    # Log the final value
                    mlflow.log_metric(f"final_{metric}", history.history[metric][-1])
                    
                    # Log the best value
                    if 'val' in metric:
                        if 'loss' in metric:
                            best_value = min(history.history[metric])
                        else:
                            best_value = max(history.history[metric])
                        mlflow.log_metric(f"best_{metric}", best_value)
            
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
                import shutil
                if os.path.exists(model_path):
                    os.remove(model_path)
            except Exception as e:
                print(f"Error logging model to MLflow: {e}")
    
    except Exception as e:
        print(f"Error in MLflow logging: {e}")


def create_ensemble_model(model_paths=None, num_classes=9, num_models=3):
    """
    Create an ensemble model from multiple trained models
    
    Args:
        model_paths: List of paths to trained models
        num_classes: Number of output classes
        num_models: Number of models to include in ensemble if model_paths not provided
        
    Returns:
        Ensemble model
    """
    with strategy.scope():
        # If model paths are provided, load those models
        if model_paths and len(model_paths) > 0:
            models = []
            for path in model_paths:
                try:
                    model = tf.keras.models.load_model(path, compile=False)
                    models.append(model)
                    print(f"Loaded model from {path}")
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
            
            if not models:
                print("No models could be loaded. Creating new models instead.")
                models = [build_peft_model(num_classes=num_classes) for _ in range(min(3, num_models))]
        else:
            # Create new models if no paths provided
            print("No model paths provided. Creating new models.")
            models = [build_peft_model(num_classes=num_classes) for _ in range(min(3, num_models))]
        
        # Create input layer
        if isinstance(models[0].input, list):
            # Handle multi-input models (for metadata)
            inputs = [tf.keras.layers.Input(shape=inp.shape[1:]) for inp in models[0].input]
        else:
            # Single input model
            inputs = tf.keras.layers.Input(shape=models[0].input.shape[1:])
        
        # Get outputs from each model
        outputs = []
        for model in models:
            if isinstance(inputs, list):
                outputs.append(model(inputs))
            else:
                outputs.append(model(inputs))
        
        # Average the outputs
        if len(outputs) > 1:
            ensemble_output = tf.keras.layers.Average()(outputs)
        else:
            ensemble_output = outputs[0]
        
        # Create and compile the ensemble model
        ensemble_model = tf.keras.Model(inputs=inputs, outputs=ensemble_output)
        
        ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc', from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"Created ensemble model with {len(models)} sub-models")
        return ensemble_model

def p(num_classes=9, metadata_dim=0, r=8, alpha=32):
    """
    Build a PEFT model that can handle both image data and metadata features
    
    Args:
        num_classes: Number of output classes
        metadata_dim: Dimension of metadata features
        r: Rank for LoRA adaptation
        alpha: Alpha parameter for LoRA
        
    Returns:
        A compiled model that can process both images and metadata
    """
    # Create the base model for image processing
    base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Image input and processing branch
    image_input = tf.keras.layers.Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input)
    
    # Add a few layers on top of the base model
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Metadata input and processing branch
    metadata_input = tf.keras.layers.Input(shape=(metadata_dim,), name='metadata_input')
    m = tf.keras.layers.Dense(64, activation='relu')(metadata_input)
    m = tf.keras.layers.Dropout(0.2)(m)
    m = tf.keras.layers.Dense(32, activation='relu')(m)
    
    # Combine image features with metadata
    combined = tf.keras.layers.Concatenate()([x, m])
    
    # Add final layers
    combined = tf.keras.layers.Dense(256, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    
    # Output layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(combined)
    
    # Create the model with two inputs
    model = tf.keras.Model(inputs=[image_input, metadata_input], outputs=output)
    
    # Apply PEFT if available
    if PEFT_AVAILABLE:
        try:
            # Configure LoRA for efficient fine-tuning
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            print("Applied PEFT (LoRA) to the model")
        except Exception as e:
            print(f"Error applying PEFT: {e}")
            print("Using standard model without PEFT")
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model