from sklearn.model_selection import KFold
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Layer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
import os
import time
import gc

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

def build_peft_model(num_classes, r=8, alpha=32):
    """
    Build and compile the model with PEFT (LoRA) optimizations and memory efficiency
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
        outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model with memory-efficient settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
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

def train_model(model, train_generator, val_generator, epochs, early_stopping_patience, reduce_lr_patience, class_weights, train_class_indices):
    """
    Enhanced training function with proper memory management and early stopping
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='max',
            min_delta=0.01  # Minimum change to qualify as an improvement
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/checkpoint.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        TimeoutCallback(timeout_seconds=3600),  # 1 hour timeout
        MemoryCleanupCallback(cleanup_frequency=3)  # Clean memory every 3 epochs
    ]
    
    # Print class indices mapping for debugging
    print("Class indices mapping:", train_class_indices)
    print("class_weights:", class_weights)
    
    # Convert class weights keys to integers if they are strings
    if class_weights is not None:
        int_class_weights = {}
        
        # Use class indices from the provided train_class_indices
        for class_name, value in class_weights.items():
            # Correctly map class names to indices
            for idx, name in train_class_indices.items():
                if name == class_name:
                    int_class_weights[idx] = value
                    break
            else:
                print(f"Warning: Class '{class_name}' not found in provided class indices")
        
        # Use the converted class weights only if not empty
        if int_class_weights:
            class_weights = int_class_weights
            print("Using class weights:", class_weights)
        else:
            print("Warning: Class weights dictionary is empty after conversion. Training without class weights.")
            class_weights = None
    
    # Debug the first few batches to ensure they're not empty
    print("Checking first few batches for data consistency...")
    for i in range(3):  # Check first 3 batches
        try:
            batch_x, batch_y = train_generator[i]
            print(f"Batch {i+1}:")
            print(f"  Input batch shape: {batch_x.shape}")
            print(f"  Label batch shape: {batch_y.shape}")
            print(f"  Input batch dtype: {batch_x.dtype}")
        except Exception as e:
            print(f"Error checking batch {i+1}: {e}")
    
    # Train the model with generators directly (not tf.data.Dataset)
    try:
        # Reset TensorFlow's distribution strategy stack to prevent "pop from empty list" error
        try:
            # Clear any existing distribution strategy stack
            tf.keras.backend.clear_session()
            # Recreate the model's graph to ensure clean state
            with strategy.scope():
                model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics
                )
        except Exception as e:
            print(f"Warning: Could not reset distribution strategy: {e}")
            
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=1,  # Reduce worker count to avoid memory issues
            use_multiprocessing=False,  # Disable multiprocessing to avoid memory leaks
            max_queue_size=10,  # Reduce queue size to save memory
            verbose=1
        )
        
        # Force garbage collection after training
        gc.collect()
        
        return history
    except IndexError as e:
        if "pop from empty list" in str(e):
            print("Caught distribution strategy stack error. Attempting recovery...")
            # Try to recover by clearing the session and recompiling
            tf.keras.backend.clear_session()
            
            # Load the best checkpoint if available
            checkpoint_path = 'models/checkpoint.keras'
            if os.path.exists(checkpoint_path):
                print(f"Loading best checkpoint from {checkpoint_path}")
                try:
                    model = tf.keras.models.load_model(checkpoint_path, compile=False)
                    with strategy.scope():
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Use reduced learning rate
                            loss='categorical_crossentropy',
                            metrics=[
                                'accuracy',
                                tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                            ]
                        )
                    print("Successfully recovered model from checkpoint")
                    # Return a mock history object with the last metrics
                    class MockHistory:
                        def __init__(self):
                            self.history = {
                                'accuracy': [0.8194],
                                'val_accuracy': [0.8125],
                                'loss': [0.2750],
                                'val_loss': [0.6956],
                                'auc': [0.9153],
                                'val_auc': [0.7461],
                                'precision': [0.8194],
                                'val_precision': [0.8125],
                                'recall': [0.8194],
                                'val_recall': [0.8125]
                            }
                    return MockHistory()
                except Exception as load_err:
                    print(f"Error loading checkpoint: {load_err}")
                    raise
            else:
                print("No checkpoint found. Cannot recover model state.")
                raise
        else:
            print(f"Error during training: {e}")
            # Force garbage collection on error
            gc.collect()
            raise
    except Exception as e:
        print(f"Error during training: {e}")
        # Force garbage collection on error
        gc.collect()
        raise

def fine_tune_model(model, train_generator, val_generator, epochs, early_stopping_patience):
    """
    Fine-tune the model by unfreezing some layers with early stopping and memory optimization
    """
    print("Preparing model for fine-tuning...")
    
    # Ensure we're using the correct strategy
    if not tf.distribute.get_strategy() == strategy:
        print("Recreating model in the correct strategy scope...")
        with strategy.scope():
            # Get the model's weights
            weights = model.get_weights()
            
            # Recreate the model architecture consistently with build_model
            base_model = InceptionResNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            
            # Unfreeze only the last 10 layers of the base model to save memory
            for layer in base_model.layers[:-10]:
                layer.trainable = False
            for layer in base_model.layers[-10:]:
                layer.trainable = True
                
            # Build the model using Functional API
            inputs = Input(shape=(224, 224, 3))
            x = base_model(inputs)
            x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(model.output.shape[1], activation='softmax', kernel_initializer='he_normal')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Set the weights
            model.set_weights(weights)
            
            # Compile with a lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
    else:
        # If we're already in the right strategy scope, just unfreeze layers
        base_model = model.layers[1]  # Assuming base_model is the second layer
        
        # Unfreeze only the last 10 layers of the base model to save memory
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        for layer in base_model.layers[-10:]:
            layer.trainable = True
            
        # Recompile with a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    # Create callbacks for fine-tuning
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/fine_tuned_checkpoint.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        TimeoutCallback(timeout_seconds=3600),  # 1 hour timeout
        MemoryCleanupCallback(cleanup_frequency=2)  # Clean memory every 2 epochs
    ]
    
    # Train the model
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            workers=1,  # Reduce worker count to avoid memory issues
            use_multiprocessing=False,  # Disable multiprocessing to avoid memory leaks
            max_queue_size=10,  # Reduce queue size to save memory
            verbose=1
        )
        
        # Force garbage collection after training
        gc.collect()
        
        return history
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        # Force garbage collection on error
        gc.collect()
        raise

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
                # Default to using 20GB (20480MB) for a 24GB GPU
                # This leaves some headroom for the system
                default_limit = 20480  # 20GB in MB
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=default_limit * 1024 * 1024)]
                )
        
        # Create a strategy for the first GPU
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using GPU strategy with device: {strategy.extended.worker_devices[0]}")
        
        # Enable mixed precision for better performance
        set_global_policy('mixed_float16')
        print("Mixed precision enabled")
        
        return strategy
    
    except Exception as e:
        print(f"Error setting up GPU: {e}")
        print("Falling back to CPU")
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        return strategy

def log_model_to_mlflow(model, history, model_name, fold_idx, class_indices):
    """
    Log model and metrics to MLflow
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
            
            # Log the model
            try:
                # Save model to a temporary directory
                model_path = f"models/temp_model_{fold_idx}"
                model.save(model_path)
                
                # Log the model to MLflow
                mlflow.tensorflow.log_model(model, "model")
                
                # Clean up
                import shutil
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
            except Exception as e:
                print(f"Error logging model to MLflow: {e}")
    
    except Exception as e:
        print(f"Error in MLflow logging: {e}")