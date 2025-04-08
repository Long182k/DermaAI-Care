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
strategy = tf.distribute.get_strategy()

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
    Build a model for Parameter-Efficient Fine-Tuning (PEFT)
    
    Args:
        num_classes: Number of classes to predict
        multi_label: Whether to use multi-label classification
    
    Returns:
        A compiled Keras model
    """
    try:
        if num_classes is None or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        
        # Set image data format explicitly
        tf.keras.backend.set_image_data_format('channels_last')
        
        # Clear any existing sessions and cache
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Get the current strategy without relying on global variable
        # This avoids issues with strategy stack management
        try:
            current_strategy = tf.distribute.get_strategy()
        except ValueError:
            # If no strategy is set, create a default one
            print("No active strategy found. Creating a default strategy.")
            if len(tf.config.list_physical_devices('GPU')) > 0:
                current_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                current_strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
                
        print(f"Building model with strategy: {current_strategy.__class__.__name__}")
        
        # Create the model with the appropriate strategy
        # The strategy scope is already properly handled by TensorFlow
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling=None  # Don't use pooling here, we'll add it manually
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Print class distribution
        print(f"Creating model for {num_classes} classes")
        print(f"Multi-label classification: {multi_label}")
        
        # Create the model using functional API
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        
        # Apply pre-processing (with normalization)
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Get base model features
        x = base_model(x, training=False)
        
        # Add global pooling with attention mechanism to prevent spatial bias
        # This helps prevent the model from focusing on specific regions
        attention = tf.keras.layers.Conv2D(1, kernel_size=1)(x)
        attention = tf.keras.layers.Activation('sigmoid')(attention)
        attention = tf.keras.layers.Multiply()([x, attention])
        x = tf.keras.layers.GlobalAveragePooling2D()(attention)
        
        # Use strong regularization to prevent overfitting to one class
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(
            512, 
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            bias_initializer='zeros'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Add another layer with smaller size
        x = tf.keras.layers.Dense(
            256, 
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=43),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            bias_initializer='zeros'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Add a class-balanced final layer with careful weight initialization
        # Essential for preventing bias toward any specific class
        if multi_label:
            # For multi-label, use sigmoid activation
            outputs = tf.keras.layers.Dense(
                num_classes, 
                activation='sigmoid',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=44),
                bias_initializer='zeros',  # Very important: start with no bias
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)
            )(x)
        else:
            # For single-label, use softmax activation with careful initialization
            # The key to preventing class bias is the bias initializer
            # We explicitly set all biases to zero initially
            outputs = tf.keras.layers.Dense(
                num_classes, 
                activation='softmax',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=44),
                bias_initializer='zeros',  # Critical for class balance
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)
            )(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNet_SkinLesion")
        
        # Define optimizer with gradient clipping to prevent extreme updates
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,  # Conservative learning rate
            clipnorm=1.0,  # Gradient clipping
            epsilon=1e-7    # For numerical stability
        )
        
        # Define metrics
        if multi_label:
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5)
            ]
            
            # Binary cross-entropy with label smoothing
            loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
                label_smoothing=0.1  # Prevent overconfidence
            )
        else:
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
            
            # For single-label, use categorical crossentropy with focal loss characteristics
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=0.1  # Prevent overconfidence
            )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        # Print model summary
        model.summary()
        
        # Print trainable parameters
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        print(f"Total params: {trainable_count + non_trainable_count:,}")
        print(f"Trainable params: {trainable_count:,}")
        print(f"Non-trainable params: {non_trainable_count:,}")
        
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

def setup_training_strategy():
    """
    Set up the training strategy for distributed training
    
    Returns:
        A TensorFlow distribution strategy
    """
    # Clear TensorFlow's default graph to avoid strategy stack issues
    tf.keras.backend.clear_session()
    
    # Check if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        # Use MirroredStrategy for multiple GPUs, OneDeviceStrategy for single GPU
        if len(gpus) > 1:
            # For multiple GPUs
            print(f"Using MirroredStrategy for {len(gpus)} GPUs")
            return tf.distribute.MirroredStrategy()
        else:
            # For single GPU
            print(f"Using OneDeviceStrategy for a single GPU: {gpus[0].name}")
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        # For CPU only
        print("No GPU found. Using CPU strategy.")
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")

def train_model(model, train_generator, val_generator, epochs=50, batch_size=32, 
                early_stopping_patience=10, reduce_lr_patience=5, 
                callbacks=None, class_weights=None, model_save_path=None,
                learning_rate=0.0001, multi_label=False):
    """
    Train the model using the specified strategy
    
    Args:
        model: The compiled Keras model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        callbacks: List of callbacks for training
        class_weights: Dictionary of class weights for imbalanced data
        model_save_path: Path to save the best model during training
        learning_rate: Initial learning rate for training
        multi_label: Whether this is a multi-label classification task
    """
    try:
        print(f"Training with batch size: {batch_size}")
        
        # Get the current strategy
        try:
            current_strategy = tf.distribute.get_strategy()
        except ValueError:
            # If no strategy is set, create a default one
            print("No active strategy found. Creating a default strategy for training.")
            if len(tf.config.list_physical_devices('GPU')) > 0:
                current_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                current_strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        
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
                ),
                # Log training metrics
                tf.keras.callbacks.CSVLogger(os.path.join(model_dir, 'training_log.csv'), append=True)
            ]
        
        # Check if the model is already compiled
        if not model.optimizer:
            print("Model not compiled, compiling now...")
            
            # Choose appropriate loss function based on multi_label flag
            if multi_label:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                    tf.keras.metrics.AUC(name='auc', multi_label=True),
                    tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                    tf.keras.metrics.Recall(name='recall', thresholds=0.5)
                ]
            else:
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
                metrics = [
                    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    tf.keras.metrics.AUC(name='auc', multi_label=False),
                    tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                    tf.keras.metrics.Recall(name='recall', thresholds=0.5)
                ]
            
            # Use gradient clipping to prevent extreme updates
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0,  # Gradient clipping
                epsilon=1e-7
            )
            
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
        
        # Analyze prediction distribution before training to check for bias
        print("\nPre-training prediction distribution check:")
        _check_prediction_distribution(model, val_generator, multi_label)
        
        # Train the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Analyze prediction distribution after training
        print("\nPost-training prediction distribution check:")
        _check_prediction_distribution(model, val_generator, multi_label)
        
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        return None

def _check_prediction_distribution(model, generator, multi_label=False):
    """Helper function to check prediction distribution for debugging bias"""
    try:
        # Make predictions on a small batch
        batch_x, batch_y = next(iter(generator))
        preds = model.predict(batch_x, verbose=0)
        
        # For one-hot encoded labels
        if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
            true_classes = np.argmax(batch_y, axis=1)
            num_classes = batch_y.shape[1]
        else:
            true_classes = batch_y
            num_classes = len(np.unique(true_classes))
        
        if multi_label:
            # For multi-label, count predictions above threshold
            pred_classes = (preds > 0.5).astype(int)
            positive_preds = np.sum(pred_classes, axis=0)
            print(f"Predictions per class (out of {len(batch_x)} samples):")
            for i in range(num_classes):
                print(f"  Class {i}: {positive_preds[i]} positive predictions")
        else:
            # For single-label, count argmax predictions
            pred_classes = np.argmax(preds, axis=1)
            class_counts = np.bincount(pred_classes, minlength=num_classes)
            print(f"Predictions per class (out of {len(batch_x)} samples):")
            for i in range(num_classes):
                pct = 100 * class_counts[i] / len(batch_x)
                print(f"  Class {i}: {class_counts[i]} predictions ({pct:.1f}%)")
                
        # Show prediction confidence
        print("\nPrediction confidence:")
        if multi_label:
            avg_conf = np.mean(preds, axis=0)
            for i in range(num_classes):
                print(f"  Class {i}: {avg_conf[i]:.4f} average confidence")
        else:
            # For categorical, check confidence of the predicted class
            max_conf = np.max(preds, axis=1)
            avg_conf = np.mean(max_conf)
            print(f"  Average confidence: {avg_conf:.4f}")
            print(f"  Min confidence: {np.min(max_conf):.4f}")
            print(f"  Max confidence: {np.max(max_conf):.4f}")
    
    except Exception as e:
        print(f"Error checking prediction distribution: {e}")

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
        # First, verify the base model's performance before fine-tuning
        print("Evaluating base model performance before fine-tuning...")
        base_eval = model.evaluate(val_generator, verbose=1)
        base_metrics = dict(zip(model.metrics_names, base_eval))
        
        # Check prediction distribution before fine-tuning
        print("\nPre-fine-tuning prediction distribution check:")
        _check_prediction_distribution(model, val_generator, multi_label)
        
        if 'accuracy' in base_metrics and base_metrics['accuracy'] < 0.5:
            print(f"WARNING: Base model accuracy ({base_metrics['accuracy']:.4f}) is too low for fine-tuning.")
            print("Fine-tuning may not improve performance. Consider retraining with a different approach.")
        
        # Get the current distribution strategy - critical for consistency
        try:
            current_strategy = tf.distribute.get_strategy()
        except ValueError:
            # If no strategy is set, create a default one
            print("No active strategy found. Creating a default strategy for fine-tuning.")
            if len(tf.config.list_physical_devices('GPU')) > 0:
                current_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                current_strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
                
        print(f"Fine-tuning with strategy: {current_strategy.__class__.__name__}")
        
        # Make a clone of the model to avoid distributed strategy issues
        fine_tuned_model = tf.keras.models.clone_model(model)
        fine_tuned_model.set_weights(model.get_weights())
        
        print("Creating fine-tuning model by selectively unfreezing layers...")
        
        # Count total layers and determine which ones to unfreeze
        total_layers = len([layer for layer in fine_tuned_model.layers if 'efficientnet' in layer.name])
        
        # Calculate how many layers to unfreeze (just the last 10%)
        layers_to_unfreeze = max(1, int(total_layers * 0.1))
        print(f"Model has {total_layers} EfficientNet layers. Unfreezing last {layers_to_unfreeze} layers.")
        
        # Keep track of layers to be fine-tuned
        fine_tuned_layers = []
        
        # VERY IMPORTANT: Keep all BatchNormalization layers FROZEN during fine-tuning
        # This prevents instability in the model
        for layer in fine_tuned_model.layers:
            # Start with everything frozen
            layer.trainable = False
            
            # Special handling for BatchNormalization
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False  # Keep frozen during fine-tuning
            
            # Unfreeze only the top dense layers - this is critical for class balance
            elif isinstance(layer, tf.keras.layers.Dense):
                # For dense layers, only unfreeze the ones before the final output layer
                if layer != fine_tuned_model.layers[-1]:  # Not the output layer
                    layer.trainable = True
                    fine_tuned_layers.append(layer.name)
        
        print(f"Layers unfrozen for fine-tuning: {fine_tuned_layers}")
        
        # Define callbacks with more patience and careful monitoring
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                min_delta=0.001  # Be more conservative about stopping
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More gentle reduction
                patience=early_stopping_patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            # Add a callback to monitor gradients and prevent extreme updates
            tf.keras.callbacks.TerminateOnNaN()
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
                    save_weights_only=False,
                    verbose=1
                )
            )
        
        # Define loss function based on multi-label setting - KEEP THE SAME AS ORIGINAL
        if multi_label:
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5)
            ]
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        
        # Compile model with MUCH smaller learning rate for fine-tuning
        # 10x-100x smaller than original learning rate
        fine_tuned_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5,  # Very small LR
                clipnorm=1.0  # Add gradient clipping
            ),
            loss=loss,
            metrics=metrics
        )
        
        # Ensure class balance is preserved during fine-tuning by using the same weights
        if class_weights is None:
            print("WARNING: No class weights provided for fine-tuning. This may lead to class imbalance issues.")
        else:
            print(f"Using class weights for fine-tuning: {class_weights}")
        
        # Train with a smaller batch size for stability
        fine_tune_batch_size = min(16, batch_size)  # Reduce batch size for stability
        
        try:
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
        
        # Check prediction distribution after fine-tuning
        print("\nPost-fine-tuning prediction distribution check:")
        _check_prediction_distribution(fine_tuned_model, val_generator, multi_label)
        
        # After fine-tuning, evaluate again to make sure we didn't degrade performance
        print("Evaluating model after fine-tuning...")
        post_eval = fine_tuned_model.evaluate(val_generator, verbose=1)
        post_metrics = dict(zip(fine_tuned_model.metrics_names, post_eval))
        
        # Compare before and after
        print("\nPerformance comparison:")
        print("-" * 50)
        for metric in base_metrics:
            if metric in post_metrics:
                change = post_metrics[metric] - base_metrics[metric]
                print(f"{metric}: {base_metrics[metric]:.4f} -> {post_metrics[metric]:.4f} (Change: {change:+.4f})")
        
        # If fine-tuning degraded performance significantly, revert to original model
        if ('accuracy' in base_metrics and 'accuracy' in post_metrics and
            post_metrics['accuracy'] < base_metrics['accuracy'] * 0.9):  # >10% worse
            print("\nWARNING: Fine-tuning degraded model performance significantly.")
            print("Reverting to original model...")
            if model_save_path:
                # Save the fine-tuned model with a different name anyway for analysis
                fine_tuned_path = model_save_path.replace('.keras', '_fine_tuned_rejected.keras')
                fine_tuned_model.save(fine_tuned_path)
                print(f"Rejected fine-tuned model saved to {fine_tuned_path} for analysis")
            return history, model  # Return original model
        
        # Save the fine-tuned model
        if model_save_path:
            fine_tuned_path = model_save_path.replace('.keras', '_fine_tuned.keras')
            fine_tuned_model.save(fine_tuned_path)
            print(f"Fine-tuned model saved to {fine_tuned_path}")
        
        return history, fine_tuned_model  # Return the fine-tuned model
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        traceback.print_exc()
        return None, model  # Return the original model in case of error

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