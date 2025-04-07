import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB5
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall
import gc
import numpy as np

# Define focal loss function to better handle class imbalance
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

# Define a function to build the EfficientNet model
def build_efficientnet_model(num_classes, model_size='B3', input_shape=(224, 224, 3), multi_label=True, dropout_rate=0.5):
    """
    Build and compile an EfficientNet model for skin lesion classification
    
    Args:
        num_classes: Number of output classes
        model_size: EfficientNet model size ('B0', 'B3', or 'B5')
        input_shape: Input image shape
        multi_label: Whether to use multi-label classification (sigmoid activation)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled EfficientNet model
    """
    # Clear any existing session
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Enable mixed precision for faster training and lower memory usage
    original_policy = tf.keras.mixed_precision.global_policy()
    set_global_policy('mixed_float16')
    
    try:
        # Create input layer
        inputs = Input(shape=input_shape)
        
        # Select the appropriate EfficientNet model based on size
        if model_size == 'B0':
            base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        elif model_size == 'B5':
            base_model = EfficientNetB5(include_top=False, weights='imagenet', input_tensor=inputs)
        else:  # Default to B3
            base_model = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        
        # Add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Add another fully-connected layer
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        if multi_label:
            outputs = Dense(num_classes, activation='sigmoid')(x)
        else:
            outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create the model
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
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    except Exception as e:
        print(f"Error building EfficientNet model: {e}")
        # Restore original policy on error
        tf.keras.mixed_precision.set_global_policy(original_policy)
        return None

# Function to fine-tune the EfficientNet model
def fine_tune_efficientnet(model, learning_rate=1e-5, fine_tune_layers=100):
    """
    Prepare the model for fine-tuning by unfreezing some layers
    
    Args:
        model: The pre-trained EfficientNet model
        learning_rate: Learning rate for fine-tuning
        fine_tune_layers: Number of layers to unfreeze from the end
        
    Returns:
        Model ready for fine-tuning
    """
    # Get the base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):
            base_model = layer
            break
    
    if base_model is None:
        print("Could not find base model for fine-tuning")
        return model
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers except the last fine_tune_layers
    total_layers = len(base_model.layers)
    for layer in base_model.layers[:total_layers - fine_tune_layers]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    if model.output_shape[-1] > 1 and model.layers[-1].activation.__name__ == 'sigmoid':
        # Multi-label classification
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            BinaryAccuracy(name='accuracy'),
            AUC(name='auc', multi_label=True),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    else:
        # Single-label classification
        loss = focal_loss(gamma=2.0, alpha=0.25)
        metrics = [
            'accuracy',
            AUC(name='auc', from_logits=False),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    # Print summary of trainable layers
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print(f"Total parameters: {trainable_count + non_trainable_count:,}")
    
    return model

# Function to create a model ensemble for better performance
def create_efficientnet_ensemble(num_classes, input_shape=(224, 224, 3), multi_label=True):
    """
    Create an ensemble of EfficientNet models with different sizes
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        multi_label: Whether to use multi-label classification
        
    Returns:
        List of compiled EfficientNet models
    """
    models = []
    
    # Create models with different architectures
    for model_size, dropout_rate in [('B0', 0.4), ('B3', 0.5), ('B5', 0.6)]:
        model = build_efficientnet_model(
            num_classes=num_classes,
            model_size=model_size,
            input_shape=input_shape,
            multi_label=multi_label,
            dropout_rate=dropout_rate
        )
        
        if model is not None:
            models.append(model)
    
    return models

# Function to make predictions with the ensemble
def predict_with_ensemble(models, x, threshold=0.5):
    """
    Make predictions using an ensemble of models
    
    Args:
        models: List of trained models
        x: Input data
        threshold: Threshold for binary classification
        
    Returns:
        Ensemble predictions
    """
    # Get predictions from each model
    all_preds = [model.predict(x) for model in models]
    
    # Average the predictions
    avg_preds = np.mean(all_preds, axis=0)
    
    # For multi-label classification, apply threshold
    if models[0].output_shape[-1] > 1 and models[0].layers[-1].activation.__name__ == 'sigmoid':
        binary_preds = (avg_preds >= threshold).astype(int)
        return binary_preds, avg_preds
    else:
        # For single-label classification, take argmax
        class_preds = np.argmax(avg_preds, axis=1)
        return class_preds, avg_preds