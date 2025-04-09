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
    Train the model with improved learning rate scheduling and regularization
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
            
            # Create a learning rate scheduler with warmup
            initial_learning_rate = learning_rate
            warmup_epochs = 5
            total_steps = epochs * len(train_generator)
            warmup_steps = warmup_epochs * len(train_generator)
            
            lr_schedule = tf.keras.optimizers.schedules.WarmUp(
                initial_learning_rate=initial_learning_rate,
                decay_schedule_fn=tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=total_steps - warmup_steps,
                    alpha=0.1
                ),
                warmup_steps=warmup_steps
            )
            
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
                # Learning rate scheduler
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: lr_schedule(epoch * len(train_generator))
                )
            ]
        
        # Compile model with improved optimizer settings
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy' if multi_label else 'sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        # Train the model
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

def fine_tune_model(model, train_generator, val_generator, epochs=5, learning_rate=5e-6, 
                   class_weights=None, callbacks=None, batch_size=32, verbose=1,
                   model_save_path=None):
    """
    Fine-tune the model with improved layer unfreezing strategy and learning rate scheduling
    """
    try:
        # Evaluate the model before fine-tuning to establish baseline
        print("Evaluating base model performance before fine-tuning...")
        base_metrics = model.evaluate(val_generator, verbose=1)
        print(f"Base model metrics: {dict(zip(model.metrics_names, base_metrics))}")
        
        # Create a new model for fine-tuning
        fine_tuned_model = tf.keras.models.clone_model(model)
        fine_tuned_model.set_weights(model.get_weights())
        
        # Print model architecture for debugging
        fine_tuned_model.summary()
        
        # Fine-tune with careful layer unfreezing
        print("Creating fine-tuning model by selectively unfreezing layers...")
        
        # Find the base model within the model architecture
        # Identify the EfficientNet base model layer
        base_model_layer = None
        for layer in fine_tuned_model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model_layer = layer
                break
        
        if base_model_layer:
            # Freeze all base model layers first
            base_model_layer.trainable = True
            
            # Freeze early layers, unfreeze later layers
            # For EfficientNet, focus on the last few blocks
            total_layers = len(base_model_layer.layers)
            for i, layer in enumerate(base_model_layer.layers):
                # Keep early layers frozen (up to 80% of the network)
                if i < int(total_layers * 0.8):
                    layer.trainable = False
                else:
                    # For BatchNormalization layers, keep them frozen during fine-tuning
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = False
                    else:
                        layer.trainable = True
        
        # Make sure all dense layers are trainable
        trainable_layers = []
        for layer in fine_tuned_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.trainable = True
                trainable_layers.append(layer.name)
        
        print(f"Layers unfrozen for fine-tuning: {trainable_layers}")
        
        if class_weights:
            print(f"Using class weights for fine-tuning: {class_weights}")
        
        # Compile with appropriate metrics and loss
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Defined metrics that won't conflict with sample_weight
        # The key issue is to use metrics that don't internally use sample_weight
        metrics = [
            # Use functions instead of string names to avoid sample_weight conflicts
            tf.keras.metrics.binary_accuracy,
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        
        fine_tuned_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )
        
        # Use smaller batch size for stability
        fine_tune_batch_size = min(16, batch_size)
        
        # Create checkpointing
        if model_save_path:
            checkpoint_path = os.path.join(os.path.dirname(model_save_path), 'fine_tuned_model_best.keras')
        else:
            checkpoint_path = 'models/fine_tuned_model_best.keras'
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Set up callbacks with better learning rate scheduling
        if callbacks is None:
            callbacks = [
                # Early stopping
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=1
                ),
                # Save best model
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                # Reduce learning rate
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        # Train with class weights but handle sample_weight conflict
        # Convert class weights to sample weights manually if needed
        if class_weights and hasattr(train_generator, 'compute_sample_weights'):
            # Use custom function if available
            sample_weights = train_generator.compute_sample_weights(class_weights)
            history = fine_tuned_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                batch_size=fine_tune_batch_size,
                verbose=verbose
            )
        else:
            # Regular training with class weights
            try:
                history = fine_tuned_model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    batch_size=fine_tune_batch_size,
                    verbose=verbose
                )
            except TypeError as e:
                # If class_weight causes an error, try without it
                print(f"Error using class_weight: {e}")
                print("Continuing fine-tuning without class weights...")
                history = fine_tuned_model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    callbacks=callbacks,
                    batch_size=fine_tune_batch_size,
                    verbose=verbose
                )
        
        # Save the fine-tuned model
        if model_save_path:
            fine_tuned_model_path = os.path.join(os.path.dirname(model_save_path), 'fine_tuned_model.keras')
            fine_tuned_model.save(fine_tuned_model_path)
            print(f"Fine-tuned model saved to: {fine_tuned_model_path}")
        
        return fine_tuned_model, history
        
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        traceback.print_exc()
        return None, None

def build_peft_model(num_classes, multi_label=False, use_focal_loss=False):
    """
    Build a PEFT model with improved architecture and class handling
    
    Args:
        num_classes: Number of output classes
        multi_label: Whether to use multi-label classification
        use_focal_loss: Whether to use focal loss for handling class imbalance
    """
    try:
        # Validate number of classes
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {num_classes}")
        print(f"Building model with {num_classes} classes, multi_label={multi_label}, focal_loss={use_focal_loss}")
        
        # Set image data format explicitly
        tf.keras.backend.set_image_data_format('channels_last')
        
        # Create base model with proper input shape
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Create attention mechanism to focus on relevant parts of the image
        input_layer = tf.keras.Input(shape=(224, 224, 3), name='input_layer_1')
        x = base_model(input_layer)
        
        # Add attention mechanism (CBAM-inspired)
        attention = tf.keras.layers.Conv2D(1, kernel_size=1)(x)
        attention = tf.keras.layers.Activation('sigmoid', name='activation')(attention)
        x = tf.keras.layers.Multiply()([x, attention])
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Batch normalization helps with internal covariate shift
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Add classification head with proper initialization
        # 512-unit dense layer
        x = tf.keras.layers.Dense(
            512,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            name='dense'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # 256-unit dense layer
        x = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            name='dense_1'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer with proper initialization
        # For multi-label, use sigmoid activation
        if multi_label:
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'
        
        outputs = tf.keras.layers.Dense(
            num_classes,
            activation=activation,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_initializer='zeros',
            name='dense_2'
        )(x)
        
        model = tf.keras.Model(inputs=input_layer, outputs=outputs, name="EfficientNet_SkinLesion")
        
        # Use Focal Loss for better handling of class imbalance when requested
        if use_focal_loss and multi_label:
            print("Using Focal Loss for multi-label classification")
            def focal_loss(gamma=2.0, alpha=0.25):
                def focal_loss_fixed(y_true, y_pred):
                    # Clip prediction values to prevent extreme loss values
                    epsilon = 1e-7
                    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                    
                    # Calculate focal loss
                    cross_entropy = -y_true * tf.math.log(y_pred)
                    weight = tf.pow(1. - y_pred, gamma) * y_true
                    loss = alpha * weight * cross_entropy
                    
                    # Sum over all classes
                    return tf.reduce_sum(loss, axis=-1)
                return focal_loss_fixed
            
            loss = focal_loss(gamma=2.0, alpha=0.25)
        elif use_focal_loss and not multi_label:
            print("Using Focal Loss for single-label classification")
            # Focal loss for categorical (single-label) classification
            def categorical_focal_loss(gamma=2.0, alpha=0.25):
                def focal_loss_fixed(y_true, y_pred):
                    # Clip prediction values
                    epsilon = 1e-7
                    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                    
                    # For sparse categorical format, convert to one-hot
                    if len(tf.shape(y_true)) == 1:
                        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
                    
                    # Calculate focal loss
                    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
                    weight = tf.pow(1. - tf.reduce_sum(y_true * y_pred, axis=-1), gamma)
                    focal_loss = alpha * weight * cross_entropy
                    return focal_loss
                return focal_loss_fixed
            
            loss = categorical_focal_loss(gamma=2.0, alpha=0.25)
        else:
            print(f"Using standard {'binary_crossentropy' if multi_label else 'sparse_categorical_crossentropy'} loss")
            
        # Compile with appropriate metrics
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=1e-4,
                weight_decay=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss=loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        traceback.print_exc()
        return None

# Add this function to help import in train.py
def get_fixed_train_functions():
    """Returns the fixed training functions"""
    return train_model, fine_tune_model, build_peft_model 