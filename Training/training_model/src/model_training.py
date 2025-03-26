from sklearn.model_selection import KFold
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Layer
import numpy as np
import os

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
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

# Custom LoRA layer for TensorFlow
class LoRALayer(Layer):
    def __init__(self, in_features, out_features, r=8, alpha=32, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        
        # Initialize A with Gaussian and B with zeros
        self.lora_A = self.add_weight(
            "lora_A", 
            shape=(self.in_features, self.r),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )
        self.lora_B = self.add_weight(
            "lora_B", 
            shape=(self.r, self.out_features),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        
        # Scaling factor
        self.scaling = self.alpha / self.r
        
    def call(self, inputs):
        # LoRA adaptation: inputs @ (A @ B) * scaling
        return tf.matmul(tf.matmul(inputs, self.lora_A), self.lora_B) * self.scaling
    
    def get_config(self):
        config = super(LoRALayer, self).get_config()
        config.update({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "r": self.r,
            "alpha": self.alpha
        })
        return config

# At the beginning of the file
from tensorflow.keras.mixed_precision import set_global_policy

# In the build_peft_model function
def build_peft_model(num_classes, r=8, alpha=32):
    """
    Build and compile the model with PEFT (LoRA) optimizations
    """
    # Enable mixed precision for faster training on L4 GPU
    set_global_policy('mixed_float16')
    
    with strategy.scope():
        # Create the base model
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
        
        # Compile the model
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

def build_model(num_classes):
    """
    Build and compile the model with memory optimizations
    """
    with strategy.scope():
        # Create the base model
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
        x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
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

def train_model(model, train_generator, val_generator, epochs, early_stopping_patience, reduce_lr_patience, class_weights=None):
    """
    Enhanced training function with proper distribution strategy and early stopping
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
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=4,
        use_multiprocessing=True
    )
    
    return history

def fine_tune_model(model, train_generator, val_generator, epochs, early_stopping_patience):
    """
    Fine-tune the model by unfreezing some layers with early stopping
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
            
            # Unfreeze the last 20 layers of the base model
            for layer in base_model.layers[-20:]:
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
        
        # Unfreeze the last 20 layers of the base model
        for layer in base_model.layers[-20:]:
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
    
    # Define callbacks for fine-tuning
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
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/fine_tuned_checkpoint.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )
    
    return history