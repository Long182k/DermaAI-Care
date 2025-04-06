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
def build_peft_model(num_classes, r=8, alpha=32, multi_label=False):
    """
    Build and compile the model with PEFT (LoRA) optimizations and memory efficiency
    
    Args:
        num_classes: Number of output classes
        r: Rank of the low-rank adaptation matrices
        alpha: Scaling factor for the adaptation
        multi_label: Whether to use multi-label classification (sigmoid activation)
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
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        else:
            loss = focal_loss(gamma=2.0, alpha=0.25)
            metrics = [
                'accuracy',
                tf.keras.metrics.AUC(name='auc', from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        # Compile the model with memory-efficient settings and proper metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss,
            metrics=metrics
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
    multi_label=False
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
    """
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
        MemoryCleanupCallback()
    ]
    
    # Update optimizer with specified learning rate
    model.optimizer.learning_rate = learning_rate
    
    # For multi-label classification with class weights, we need to handle it differently
    if multi_label and class_weights:
        # Convert class weights to sample weights for multi-label
        print(f"Applying class weights for multi-label classification: {class_weights}")
        
        # Create a weighted binary crossentropy loss
        def weighted_binary_crossentropy(y_true, y_pred):
            # Create per-class weights tensor
            class_weights_tensor = tf.constant(
                [class_weights.get(i, 1.0) for i in range(y_true.shape[-1])],
                dtype=tf.float32
            )
            
            # Calculate binary crossentropy
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            
            # Apply weights based on the true classes in each sample
            # Use broadcasting to apply weights to each sample/class
            weights = tf.reduce_sum(y_true * class_weights_tensor, axis=-1)
            
            # Reshape weights to match bce shape for broadcasting
            weights = tf.reshape(weights, [-1, 1])
            
            # Calculate weighted loss - use direct multiplication
            weighted_loss = bce * weights
            
            # Return mean loss
            return tf.reduce_mean(weighted_loss)
        
        # Recompile the model with our custom loss
        metrics = model.metrics
        optimizer = model.optimizer
        model.compile(
            optimizer=optimizer,
            loss=weighted_binary_crossentropy,
            metrics=metrics
        )
    
    # Train the model
    try:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=None,  # Always use None, we handle weights in custom loss
            verbose=1
        )
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
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
    class_weights=None
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
    """
    print("Starting fine-tuning process...")
    
    # Clear session to avoid strategy mixing issues
    tf.keras.backend.clear_session()
    
    # Create a clone of the model to avoid strategy mixing issues
    model_config = model.get_config()
    weights = model.get_weights()
    
    # Recreate the model with the current strategy
    with strategy.scope():
        new_model = tf.keras.Model.from_config(model_config)
        new_model.set_weights(weights)
        
        # Unfreeze the last few layers of the base model
        base_model = new_model.layers[1]  # Get the InceptionResNetV2 model
        for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
            layer.trainable = True
        
        # For multi-label classification with class weights, we need a custom loss
        if multi_label and class_weights:
            # Create a weighted binary crossentropy loss
            def weighted_binary_crossentropy(y_true, y_pred):
                # Create per-class weights tensor
                class_weights_tensor = tf.constant(
                    [class_weights.get(i, 1.0) for i in range(y_true.shape[-1])],
                    dtype=tf.float32
                )
                
                # Calculate binary crossentropy
                bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                
                # Apply weights based on the true classes in each sample
                weights = tf.reduce_sum(y_true * class_weights_tensor, axis=-1)
                
                # Reshape weights to match bce shape for broadcasting
                weights = tf.reshape(weights, [-1, 1])
                
                # Calculate weighted loss
                weighted_loss = bce * weights
                
                # Return mean loss
                return tf.reduce_mean(weighted_loss)
            
            loss = weighted_binary_crossentropy
            print("Using weighted binary crossentropy for multi-label fine-tuning")
        elif multi_label:
            loss = tf.keras.losses.BinaryCrossentropy()
            print("Using binary crossentropy for multi-label fine-tuning")
        else:
            loss = focal_loss(gamma=2.0, alpha=0.25)
            print("Using focal loss for single-label fine-tuning")
        
        # Set up appropriate metrics based on task type
        if multi_label:
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        else:
            metrics = [
                'accuracy',
                tf.keras.metrics.AUC(name='auc', from_logits=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        # Compile the model with appropriate loss and metrics
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
            loss=loss,
            metrics=metrics
        )
    
    # Set up callbacks for fine-tuning
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
        TimeoutCallback(timeout_seconds=3600),  # 1 hour timeout
        MemoryCleanupCallback()
    ]
    
    # Fine-tune the model
    try:
        history = new_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=None,  # We handle weights in the loss function
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
                metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
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
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    
    return ensemble_model, models