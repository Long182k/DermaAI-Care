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
        
        # Get predictions
        preds = model.predict(x_test)
        
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

def build_peft_model(num_classes, multi_label=False, use_focal_loss=False):
    """
    Build an EfficientNet model with Parameter-Efficient Fine-Tuning (PEFT) approach.
    
    Args:
        num_classes: Number of classes for classification
        multi_label: Whether to use multi-label classification
        use_focal_loss: Whether to use focal loss for class imbalance
        
    Returns:
        Compiled model
    """
    try:
        # Validate inputs and set parameters
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {num_classes}")
        
        print(f"\nBuilding EfficientNet model with {num_classes} classes")
        print(f"Multi-label: {multi_label}, Using focal loss: {use_focal_loss}")
        
        # Set image data format for consistency
        K = tf.keras.backend
        K.set_image_data_format('channels_last')
        
        # Define input shape - standard for medical imaging tasks
        input_shape = (224, 224, 3)
        
        # Create input layer with explicit name
        inputs = tf.keras.layers.Input(shape=input_shape, name='model_input')
        
        # Add preprocessing layer (normalize pixel values)
        x = tf.keras.layers.Rescaling(1./255)(inputs)
        
        # Load EfficientNetB0 as base model without top layers and weights from ImageNet
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None  # No pooling here, we'll add it later
        )
        
        # Freeze base model for initial training
        base_model.trainable = False
        
        # Apply the base model to our input
        x = base_model(x)
        
        # Add global average pooling to reduce parameters
        x = tf.keras.layers.GlobalAveragePooling2D(name='pooling_layer')(x)
        
        # Add dropout for regularization
        x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
        
        # Add a dense layer with batch normalization and activation
        x = tf.keras.layers.Dense(512, name='dense_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
        x = tf.keras.layers.Activation('relu', name='act_1')(x)
        
        # Add more dropout for better regularization
        x = tf.keras.layers.Dropout(0.4, name='dropout_2')(x)
        
        # Output layer with special initialization to prevent class collapse
        if multi_label:
            # For multi-label, use sigmoid activations (independent classes)
            # Use special initial bias to prevent class collapse
            # Calculate initial bias based on assumed class frequency
            # Usually rare classes in dermatology datasets
            initial_bias = np.log([0.05] * num_classes)  # Assume 5% prevalence for each class
            
            outputs = tf.keras.layers.Dense(
                num_classes, 
                activation='sigmoid',
                name='output',
                bias_initializer=tf.keras.initializers.Constant(initial_bias),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
            )(x)
            
            # For multi-label, always use binary cross-entropy
            if use_focal_loss:
                def focal_loss(gamma=2., alpha=4.):
                    def focal_loss_fixed(y_true, y_pred):
                        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
                        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
                        
                        # Clip to prevent NaN's and Inf's
                        pt_1 = tf.clip_by_value(pt_1, 1e-7, 1.0 - 1e-7)
                        pt_0 = tf.clip_by_value(pt_0, 1e-7, 1.0 - 1e-7)
                        
                        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
                    return focal_loss_fixed
                
                loss = focal_loss(gamma=2, alpha=0.8)
                print("Using Focal Loss for multi-label classification")
            else:
                loss = 'binary_crossentropy'
                print("Using Binary Cross Entropy for multi-label classification")
        else:
            # For single-label, use softmax activation (mutually exclusive classes)
            # Apply special initialization strategy to prevent dominant class issue
            outputs = tf.keras.layers.Dense(
                num_classes, 
                activation='softmax',
                name='output',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                # Balanced initial bias for all classes
                bias_initializer=tf.keras.initializers.Constant(0.0)
            )(x)
            
            # For single-label classification
            if use_focal_loss and num_classes > 1:
                def categorical_focal_loss(gamma=2., alpha=.25):
                    """
                    Categorical version of Focal Loss for addressing class imbalance.
                    :param gamma: Focusing parameter for modulating loss for hard examples
                    :param alpha: Balancing parameter for addressing class imbalance
                    """
                    def categorical_focal_loss_fixed(y_true, y_pred):
                        # Clip prediction values to prevent log(0) errors
                        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                        
                        # Standard cross-entropy calculation
                        cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
                        
                        # Calculate focal weights
                        # When y_true is one-hot encoded, this picks out the predicted prob of the true class
                        probs = tf.reduce_sum(y_true * y_pred, axis=-1)
                        
                        # Apply gamma focusing parameter
                        focal_weights = tf.pow(1.0 - probs, gamma)
                        
                        # Apply the weights to cross-entropy
                        weighted_cross_entropy = focal_weights * cross_entropy
                        
                        return tf.reduce_mean(weighted_cross_entropy)
                    
                    return categorical_focal_loss_fixed
                
                loss = categorical_focal_loss(gamma=2.0, alpha=0.25)
                print("Using Categorical Focal Loss for single-label classification")
            else:
                loss = 'categorical_crossentropy'
                print("Using Categorical Cross Entropy for single-label classification")
        
        # Compile metrics - ensure they are function references, not strings
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy') if multi_label else tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc', multi_label=multi_label),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        
        # Create and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNet_PEFT")
        
        # Compile with appropriate optimizer and loss
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=1e-3,
                weight_decay=1e-5,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss=loss,
            metrics=metrics
        )
        
        # Print model summary
        model.summary()
        
        print(f"Model built successfully with {model.count_params():,} parameters")
        print(f"Base model frozen with {base_model.count_params():,} parameters")
        
        return model
        
    except Exception as e:
        print(f"Error in build_peft_model: {str(e)}")
        traceback.print_exc()
        # Add fully connected layer with careful initialization
        x = tf.keras.layers.Dense(
            512,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=42),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            name='fc_layer'
        )(x)
        
        # Add batch normalization for training stability
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Add another dropout layer
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Output layer with neutral bias initialization (critical to avoid class collapse)
        if multi_label:
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'
        
        # Set bias initialization to zeros to avoid class collapse
        outputs = tf.keras.layers.Dense(
            num_classes,
            activation=activation,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            bias_initializer='zeros',  # Neutral bias to prevent collapse to dominant class
            name='output'
        )(x)
        
        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNet_SkinLesion")
        
        # Define focal loss if requested
        if use_focal_loss:
            print(f"Using Focal Loss for {'multi' if multi_label else 'single'}-label classification")
            
            def focal_loss(gamma=2.0, alpha=0.25):
                def focal_loss_fixed(y_true, y_pred):
                    epsilon = 1e-7
                    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                    
                    # For sparse categorical format (single-label), convert to one-hot
                    if not multi_label and len(tf.shape(y_true)) == 1:
                        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
                    
                    # Calculate cross entropy
                    cross_entropy = -y_true * tf.math.log(y_pred)
                    
                    # Calculate focal weight
                    p_t = tf.exp(-cross_entropy)
                    # Add the alpha weighing factor
                    alpha_factor = y_true * alpha + (1-y_true) * (1-alpha)
                    focal_weight = alpha_factor * tf.pow((1-p_t), gamma)
                    
                    # Calculate focal loss
                    focal_loss = focal_weight * cross_entropy
                    
                    # Sum over classes and average over samples
                    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
                
                return focal_loss_fixed
            
            loss = focal_loss(gamma=2.0, alpha=0.25)
            
        # Compile model with appropriate optimizer and metrics
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Separate function references for metrics to avoid sample_weight issues
        accuracy_metric = tf.keras.metrics.BinaryAccuracy() if multi_label else tf.keras.metrics.SparseCategoricalAccuracy()
        auc_metric = tf.keras.metrics.AUC(multi_label=multi_label, name='auc')
        precision_metric = tf.keras.metrics.Precision(name='precision')
        recall_metric = tf.keras.metrics.Recall(name='recall')
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                accuracy_metric,
                auc_metric,
                precision_metric,
                recall_metric
            ]
        )
        
        # Print model summary for debugging
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        traceback.print_exc()
        return None

# Add this function to help import in train.py
def get_fixed_train_functions():
    """Returns the fixed training functions"""
    return train_model, fine_tune_model, build_peft_model 