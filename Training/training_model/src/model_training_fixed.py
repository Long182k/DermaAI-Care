import tensorflow as tf
import numpy as np
import os
import traceback
import gc

# Add focal loss implementation
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss function for handling class imbalance in multi-class classification
    
    Args:
        gamma: Focusing parameter that controls how much to focus on hard examples
        alpha: Weighting factor for rare classes
        
    Returns:
        A loss function that can be used with Keras models
    """
    def loss_function(y_true, y_pred):
        # Convert one-hot encoded y_true to class indices if needed
        if len(tf.shape(y_true)) > 1 and tf.shape(y_true)[1] > 1:
            y_true = tf.argmax(y_true, axis=1)
        
        # Convert to one-hot encoding for focal loss calculation
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[1])
        
        # Calculate focal loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true_one_hot, 
            logits=y_pred
        )
        
        # Apply focusing parameter
        p_t = tf.exp(-cross_entropy)
        focal_loss = alpha * tf.pow(1 - p_t, gamma) * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    
    return loss_function

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

def fine_tune_model(model, train_generator, val_generator, diagnosis_to_idx, epochs=5, batch_size=32, class_weights=None, early_stopping=10, reduce_lr=5, fine_tune_lr=1e-5, callbacks=None):
    """
    Fine-tune a pre-trained model for skin lesion classification with careful layer unfreezing
    
    Args:
        model: Pre-trained model to fine-tune
        train_generator: Generator for training data
        val_generator: Generator for validation data
        diagnosis_to_idx: Mapping of diagnosis names to indices
        epochs: Number of epochs for fine-tuning
        batch_size: Batch size for training
        class_weights: Weights for each class to handle imbalance
        early_stopping: Number of epochs to wait before early stopping
        reduce_lr: Number of epochs to wait before reducing learning rate
        fine_tune_lr: Learning rate for fine-tuning (default: 1e-5)
        callbacks: List of callbacks for model training
    
    Returns:
        Fine-tuned model
    """
    try:
        print("\n=== Starting Fine-tuning Process ===")
        
        # === First, evaluate the model to see its baseline performance ===
        print("\nEvaluating model before fine-tuning...")
        eval_results = model.evaluate(val_generator, verbose=1)
        metric_names = model.metrics_names
        print("Baseline model performance:")
        for name, value in zip(metric_names, eval_results):
            print(f" - {name}: {value:.4f}")
        
        # === Check for prediction bias ===
        print("\nAnalyzing prediction distribution before fine-tuning...")
        
        # Get a batch of validation data
        if isinstance(val_generator, tf.data.Dataset):
            # For tf.data.Dataset, iterate to get the first batch
            for x_test, y_test in val_generator.take(1):
                break
        else:
            # For other generators, use indexing
            x_test, y_test = val_generator[0]
        
        # Get predictions - handle the input properly
        try:
            preds = model.predict(x_test, verbose=0)  # Add verbose=0 to reduce output and only pass x_test
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            traceback.print_exc()
            preds = None
            return model
        
        # Display class distribution
        if len(preds.shape) > 1 and preds.shape[1] > 1:  # Multi-class
            class_preds = np.argmax(preds, axis=1)
            class_counts = np.bincount(class_preds, minlength=len(diagnosis_to_idx))
            class_names = {v: k for k, v in diagnosis_to_idx.items()}
            print("Prediction class distribution:")
            for i, count in enumerate(class_counts):
                class_name = class_names.get(i, f"Class {i}")
                print(f" - {class_name}: {count} predictions ({count/len(class_preds)*100:.1f}%)")
            
            # Check if predictions are biased to one class
            max_class_ratio = np.max(class_counts) / len(class_preds)
            if max_class_ratio > 0.9:
                print(f"WARNING: Model predictions are biased towards a single class ({max_class_ratio:.2f} ratio)")
                print("Will apply extra class balancing during fine-tuning")
        
        # === Prepare for fine-tuning ===
        # Create enhanced class weights with stronger emphasis on rare classes
        if class_weights is not None:
            print("\nEnhancing class weights for fine-tuning...")
            enhanced_weights = {}
            for class_idx, weight in class_weights.items():
                # Further emphasize rare classes during fine-tuning
                # Formula: enhanced_weight = original_weight^0.75 * 1.5 if weight > 1
                if weight > 1:
                    enhanced_weights[class_idx] = weight**0.75 * 1.5
                else:
                    enhanced_weights[class_idx] = weight
            
            # Print the enhanced weights
            original_sum = sum(class_weights.values())
            enhanced_sum = sum(enhanced_weights.values())
            print("Original vs Enhanced class weights:")
            for class_idx in class_weights:
                class_name = {v: k for k, v in diagnosis_to_idx.items()}.get(class_idx, f"Class {class_idx}")
                print(f" - {class_name}: {class_weights[class_idx]:.2f} → {enhanced_weights[class_idx]:.2f}")
            
            # Use the enhanced weights for fine-tuning
            class_weights = enhanced_weights
        
        # === Set up fine-tuning of the model ===
        print("\nPreparing the model for fine-tuning...")
        
        # Carefully unfreeze layers for fine-tuning
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):  # Find the base model
                base_model = layer
                break
        
        if base_model is not None:
            print(f"Found base model: {base_model.name}")
            
            # First, ensure all layers are still frozen
            base_model.trainable = True
            
            # Count total layers in base model
            total_layers = len(base_model.layers)
            print(f"Total layers in base model: {total_layers}")
            
            # Only unfreeze the last 30% of layers (common strategy for EfficientNet)
            # This is generally more effective than unfreezing everything
            layers_to_unfreeze = int(total_layers * 0.3)
            
            # Keep early layers frozen, unfreeze later layers
            for i, layer in enumerate(base_model.layers):
                if i < (total_layers - layers_to_unfreeze):
                    layer.trainable = False
                else:
                    layer.trainable = True
                    print(f"Unfreezing layer: {layer.name}")
        else:
            # If we can't find the base model, carefully unfreeze layers 
            # Check if there are trainable dense layers
            dense_layers = [layer for layer in model.layers if 'dense' in layer.name.lower()]
            if dense_layers:
                print(f"Unfreezing dense layers for fine-tuning")
                for layer in model.layers:
                    if 'dense' in layer.name.lower() or 'batch_normalization' in layer.name.lower():
                        layer.trainable = True
                        print(f"Unfreezing layer: {layer.name}")
                    else:
                        layer.trainable = False
            else:
                # If no specific layers found, set all to trainable as a fallback
                print("No specific layers found for selective unfreezing. Setting model to trainable.")
                model.trainable = True
        
        # Print number of trainable parameters
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Non-trainable parameters: {non_trainable_count:,}")
        
        # === Recompile model with a lower learning rate for fine-tuning ===
        print(f"\nRecompiling model with fine-tuning learning rate: {fine_tune_lr}")
        
        # Get the current optimizer, loss, and metrics to preserve them
        current_optimizer = model.optimizer
        if hasattr(current_optimizer, 'learning_rate'):
            # Create a new AdamW optimizer with the fine-tuning learning rate
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=fine_tune_lr,
                weight_decay=1e-6,  # Lower weight decay for fine-tuning
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            print(f"Created new AdamW optimizer with learning rate {fine_tune_lr}")
        else:
            # Fallback optimizer if needed
            optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
            print(f"Created new Adam optimizer with learning rate {fine_tune_lr}")
        
        # Recompile with same loss and metrics but new optimizer
        model.compile(
            optimizer=optimizer,
            loss=model.loss,
            metrics=model.compiled_metrics._metrics  # Keep same metrics
        )
        
        # === Setup callbacks for fine-tuning ===
        fine_tune_callbacks = []
        
        # Add early stopping with higher patience
        fine_tune_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Add ReduceLROnPlateau with more aggressive reduction
        fine_tune_callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # More aggressive reduction
                patience=reduce_lr,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # Add custom user callbacks if provided
        if callbacks:
            fine_tune_callbacks.extend(callbacks)
        
        # === Fine-tune the model ===
        print("\nStarting fine-tuning...")
        
        # Adjust batch size if needed (smaller batch size for fine-tuning)
        if batch_size > 16:
            effective_batch_size = 16  # Limit batch size for stable fine-tuning
            print(f"Adjusting batch size from {batch_size} to {effective_batch_size} for stable fine-tuning")
        else:
            effective_batch_size = batch_size
        
        # Start fine-tuning
        fine_tune_history = None
        try:
            fine_tune_history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                class_weight=class_weights,
                callbacks=fine_tune_callbacks,
                batch_size=effective_batch_size,
                verbose=1
            )
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
            traceback.print_exc()
        
        # === Evaluate the fine-tuned model ===
        print("\nEvaluating fine-tuned model...")
        final_eval = model.evaluate(val_generator, verbose=1)
        print("Fine-tuned model performance:")
        for name, value in zip(metric_names, final_eval):
            print(f" - {name}: {value:.4f}")
        
        # === Check for prediction bias again ===
        print("\nAnalyzing prediction distribution after fine-tuning...")
        
        # Use the same test batch we extracted earlier
        new_preds = model.predict(x_test)
        
        # Display class distribution again
        if len(new_preds.shape) > 1 and new_preds.shape[1] > 1:  # Multi-class
            class_preds = np.argmax(new_preds, axis=1)
            class_counts = np.bincount(class_preds, minlength=len(diagnosis_to_idx))
            class_names = {v: k for k, v in diagnosis_to_idx.items()}
            print("Prediction class distribution after fine-tuning:")
            for i, count in enumerate(class_counts):
                class_name = class_names.get(i, f"Class {i}")
                print(f" - {class_name}: {count} predictions ({count/len(class_preds)*100:.1f}%)")
        
        # Return the fine-tuned model
        return model
        
    except Exception as e:
        print(f"Error in fine_tune_model: {str(e)}")
        traceback.print_exc()
        return model

def build_peft_model(num_classes=9, multi_label=False, use_focal_loss=True, 
                    initial_bias=-0.5, extra_regularization=False):
    """
    Build a model with Parameter-Efficient Fine-Tuning capabilities
    
    Args:
        num_classes: Number of output classes
        multi_label: Whether to use multi-label classification
        use_focal_loss: Whether to use focal loss for handling class imbalance
        initial_bias: Initial bias value for output layer
        extra_regularization: Whether to add extra L2 regularization
        
    Returns:
        A compiled model
    """
    # Create the base model - using EfficientNetB0 instead of InceptionResNetV2 to avoid GPU MaxPool gradient errors
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create input layer
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    
    # Apply preprocessing - use EfficientNet preprocessing to match the base model
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Pass through the base model
    x = base_model(x)
    
    # Add a few layers on top of the base model
    # Add regularization if requested
    reg = tf.keras.regularizers.l2(0.001) if extra_regularization else None
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer with appropriate activation
    if multi_label:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    
    # Apply initial bias to output layer
    output_bias = None
    if initial_bias is not None:
        output_bias = tf.keras.initializers.Constant(initial_bias)
    
    outputs = tf.keras.layers.Dense(
        num_classes, 
        activation=activation,
        bias_initializer=output_bias,
        kernel_regularizer=reg
    )(x)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with appropriate loss function
    if use_focal_loss:
        loss = focal_loss(gamma=2.0, alpha=0.25)
    elif multi_label:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

# Add this function to help import in train.py
def get_fixed_train_functions():
    """Returns the fixed training functions"""
    return train_model, fine_tune_model, build_peft_model, focal_loss