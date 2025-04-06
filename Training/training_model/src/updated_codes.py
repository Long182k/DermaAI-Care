def create_balanced_data_generator(generator, batch_size=64):
    """
    Creates a balanced data generator by oversampling minority classes
    
    Args:
        generator: Original data generator
        batch_size: Batch size for the new generator
        
    Returns:
        A balanced data generator
    """
    try:
        # Check if we're dealing with a YOLODetectionGenerator
        is_yolo_generator = hasattr(generator, '__class__') and 'YOLODetectionGenerator' in generator.__class__.__name__
        
        if is_yolo_generator:
            print("Detected YOLODetectionGenerator. Using specialized handling.")
            
            # For YOLODetectionGenerator, we need to handle it differently
            # First, determine the image shape from a sample batch
            try:
                sample_batch = generator[0]
                if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                    # Typical structure: ([images, metadata], labels)
                    inputs, _ = sample_batch
                    if isinstance(inputs, list) and len(inputs) > 0:
                        image_shape = inputs[0].shape[1:]  # Get shape excluding batch dimension
                    else:
                        image_shape = inputs.shape[1:]  # Get shape excluding batch dimension
                else:
                    # Default shape if structure is unexpected
                    image_shape = (224, 224, 3)
                
                print(f"Determined image shape: {image_shape}")
            except Exception as e:
                print(f"Error determining image shape: {e}")
                # Default shape
                image_shape = (224, 224, 3)
                print(f"Using default image shape: {image_shape}")
            
            # Determine number of classes
            try:
                if hasattr(generator, 'num_classes'):
                    num_classes = generator.num_classes
                elif hasattr(generator, 'diagnosis_to_idx'):
                    num_classes = len(generator.diagnosis_to_idx)
                else:
                    # Try to infer from a sample batch
                    _, labels = generator[0]
                    if len(labels.shape) > 1:
                        num_classes = labels.shape[1]
                    else:
                        # Estimate from max label value
                        num_classes = np.max(labels) + 1
                
                print(f"Determined number of classes: {num_classes}")
            except Exception as e:
                print(f"Error determining number of classes: {e}")
                # Default to 8 classes (from the error log)
                num_classes = 8
                print(f"Using default number of classes: {num_classes}")
            
            # For YOLO generator, just return the original generator
            # since we can't easily balance it without more information
            print("Returning original YOLODetectionGenerator as balancing is not supported")
            return generator
        
        # Get class distribution for standard generators
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
        
        # For standard generators, determine image shape
        if not hasattr(generator, 'image_shape'):
            if hasattr(generator, 'target_size'):
                # Standard ImageDataGenerator
                image_shape = generator.target_size + (3,)
            elif hasattr(generator, 'df') and 'image_path' in generator.df.columns:
                # Try to get image shape from the first image
                try:
                    sample_img = Image.open(generator.df['image_path'].iloc[0])
                    image_shape = sample_img.size + (3,)  # Width, Height, Channels
                except:
                    # Default shape if can't determine
                    image_shape = (224, 224, 3)
            else:
                # Default shape
                image_shape = (224, 224, 3)
        else:
            image_shape = generator.image_shape
        
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
            tf.TensorSpec(shape=(None,) + tuple(image_shape), dtype=tf.float32),
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
        traceback.print_exc() 
        return generator


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
    
    # Check if model is None (training failed)
    if model is None:
        print("Cannot fine-tune: model is None (previous training failed)")
        return None
    
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
    
    # Check if we're dealing with a YOLODetectionGenerator
    is_yolo_generator = hasattr(train_generator, '__class__') and 'YOLODetectionGenerator' in train_generator.__class__.__name__
    
    if is_yolo_generator:
        print("Detected YOLODetectionGenerator. Using specialized handling for fine-tuning.")
        
        # For YOLODetectionGenerator, we need to convert it to a format that works with model.fit
        try:
            # Create a dataset from the generator that Keras can understand
            def create_dataset_from_yolo_generator(generator):
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
                                if isinstance(batch, tuple) and len(batch) == 2:
                                    inputs, labels = batch
                                    if isinstance(inputs, list) and len(inputs) > 0:
                                        # Extract just the images from [images, metadata]
                                        yield inputs[0], labels
                                    else:
                                        # Already in correct format
                                        yield inputs, labels
                                else:
                                    # Skip incorrect structure
                                    print(f"Skipping batch {i} - incorrect structure")
                                    continue
                        except Exception as e:
                            print(f"Error in batch {i}: {e}")
                            continue
                
                # Create dataset with appropriate output signature
                if has_metadata:
                    if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                        inputs, labels = sample_batch
                        if isinstance(inputs, list) and len(inputs) >= 2:
                            images, metadata = inputs[0], inputs[1]
                            output_signature = (
                                (
                                    tf.TensorSpec(shape=images.shape, dtype=tf.float32),
                                    tf.TensorSpec(shape=metadata.shape, dtype=tf.float32)
                                ),
                                tf.TensorSpec(shape=labels.shape, dtype=tf.float32)
                            )
                        else:
                            # Fallback to expected shapes
                            print("Using fallback shapes for metadata model")
                            output_signature = (
                                (
                                    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                                    tf.TensorSpec(shape=(None, 10), dtype=tf.float32)  # Assuming 10 metadata features
                                ),
                                tf.TensorSpec(shape=(None, 8), dtype=tf.float32)  # 8 classes from error log
                            )
                    else:
                        # Fallback to expected shapes
                        print("Using fallback shapes for metadata model (tuple structure issue)")
                        output_signature = (
                            (
                                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                                tf.TensorSpec(shape=(None, 10), dtype=tf.float32)  # Assuming 10 metadata features
                            ),
                            tf.TensorSpec(shape=(None, 8), dtype=tf.float32)  # 8 classes from error log
                        )
                else:
                    if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                        inputs, labels = sample_batch
                        if isinstance(inputs, list) and len(inputs) > 0:
                            images = inputs[0]
                        else:
                            images = inputs
                        output_signature = (
                            tf.TensorSpec(shape=images.shape, dtype=tf.float32),
                            tf.TensorSpec(shape=labels.shape, dtype=tf.float32)
                        )
                    else:
                        # Fallback to expected shapes
                        print("Using fallback shapes for image-only model")
                        output_signature = (
                            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                            tf.TensorSpec(shape=(None, 8), dtype=tf.float32)  # 8 classes from error log
                        )
                
                # Create and return the dataset
                dataset = tf.data.Dataset.from_generator(
                    gen_fn,
                    output_signature=output_signature
                )
                
                return dataset.prefetch(tf.data.AUTOTUNE)
            
            # Convert generators to datasets
            train_dataset = create_dataset_from_yolo_generator(train_generator)
            val_dataset = create_dataset_from_yolo_generator(val_generator)
        except Exception as e:
            print(f"Error creating datasets from YOLO generators: {e}")
            print("Fine-tuning could not be completed. Returning original model.")
            return model
    
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
