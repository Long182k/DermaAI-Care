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
