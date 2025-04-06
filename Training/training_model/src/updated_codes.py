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
        
        # Apply PEFT if available
        if PEFT_AVAILABLE:
            try:
                # Configure LoRA for efficient fine-tuning
                # Specify target_modules to fix the error - use specific layer names
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    inference_mode=False,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=0.1,
                    target_modules=["dense_1", "lora_adaptation", "predictions"]  # Target specific layers by name
                )
                model = get_peft_model(model, peft_config)
                print("Applied PEFT (LoRA) to the model")
            except Exception as e:
                print(f"Error applying PEFT: {e}")
                print("Using standard model without PEFT")
        
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
        # Check if we're dealing with a YOLODetectionGenerator before deciding on training approach
        is_yolo_generator = hasattr(train_data, '__class__') and 'YOLODetectionGenerator' in train_data.__class__.__name__
        
        if is_yolo_generator:
            print("Detected YOLODetectionGenerator. Converting to tf.data.Dataset for compatibility")
            
            # Create a dataset from the generator that Keras can understand
            def create_dataset_from_yolo_generator(generator, batch_size):
                # Get a sample batch to determine shapes
                sample_batch = generator[0]
                
                # Determine if this is a metadata model
                has_metadata = isinstance(model.input, list) and len(model.input) > 1
                
                # Create a generator function that yields batches in the right format
                def gen_fn():
                    for i in range(len(generator)):
                        try:
                            batch = generator[i]
                            if has_metadata:
                                # For models with metadata, return the full structure
                                yield batch
                            else:
                                # For models without metadata, extract just the images
                                [images, _], labels = batch
                                yield images, labels
                        except Exception as e:
                            print(f"Error in batch {i}: {e}")
                            # Return a dummy batch with correct shapes
                            if has_metadata:
                                [images, metadata], labels = sample_batch
                                yield [np.zeros_like(images), np.zeros_like(metadata)], np.zeros_like(labels)
                            else:
                                [images, _], labels = sample_batch
                                yield np.zeros_like(images), np.zeros_like(labels)
                
                # Create dataset with appropriate output signature
                if has_metadata:
                    [images, metadata], labels = sample_batch
                    output_signature = (
                        (
                            tf.TensorSpec(shape=images.shape, dtype=tf.float32),
                            tf.TensorSpec(shape=metadata.shape, dtype=tf.float32)
                        ),
                        tf.TensorSpec(shape=labels.shape, dtype=tf.float32)
                    )
                else:
                    [images, _], labels = sample_batch
                    output_signature = (
                        tf.TensorSpec(shape=images.shape, dtype=tf.float32),
                        tf.TensorSpec(shape=labels.shape, dtype=tf.float32)
                    )
                
                # Create and return the dataset
                dataset = tf.data.Dataset.from_generator(
                    gen_fn,
                    output_signature=output_signature
                )
                
                return dataset.prefetch(tf.data.AUTOTUNE)
            
            # Convert generators to datasets
            train_dataset = create_dataset_from_yolo_generator(train_data, args.batch_size if 'args' in globals() else 16)
            val_dataset = create_dataset_from_yolo_generator(val_data, args.batch_size if 'args' in globals() else 16)
            
            # Train with the datasets
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        elif use_datasets:
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
            # For standard Keras data generators
            print("Using standard data generators for training")
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
        import traceback
        traceback.print_exc()
        
        # Force garbage collection on error
        gc.collect()
        
        return None
