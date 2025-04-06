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
            # For Keras data generators and custom generators
            print("Using data generators for training")
            
            # Check if we're dealing with a YOLODetectionGenerator
            if hasattr(train_data, '__class__') and 'YOLODetectionGenerator' in train_data.__class__.__name__:
                print("Detected YOLODetectionGenerator. Converting to tf.data.Dataset")
                
                # Create a function to yield batches from the generator
                def generator_fn():
                    for i in range(len(train_data)):
                        yield train_data[i]
                
                # Create a function to yield batches from validation generator
                def val_generator_fn():
                    for i in range(len(val_data)):
                        yield val_data[i]
                
                # Determine output shapes and types from a sample batch
                x_batch, y_batch = train_data[0]
                
                # Create tf.data.Dataset from the generator
                train_dataset = tf.data.Dataset.from_generator(
                    generator_fn,
                    output_signature=(
                        tf.TensorSpec(shape=(None,) + x_batch.shape[1:], dtype=tf.float32),
                        tf.TensorSpec(shape=(None,) + y_batch.shape[1:], dtype=tf.float32)
                    )
                ).prefetch(tf.data.AUTOTUNE)
                
                # Create validation dataset
                val_dataset = tf.data.Dataset.from_generator(
                    val_generator_fn,
                    output_signature=(
                        tf.TensorSpec(shape=(None,) + x_batch.shape[1:], dtype=tf.float32),
                        tf.TensorSpec(shape=(None,) + y_batch.shape[1:], dtype=tf.float32)
                    )
                ).prefetch(tf.data.AUTOTUNE)
                
                # Train with the datasets
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                    class_weight=class_weights if not use_focal_loss else None,
                    verbose=1
                )
            # Check if we need to create a balanced generator
            elif hasattr(train_data, 'class_counts') and sum(train_data.class_counts.values()) > 0:
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


def log_model_to_mlflow(model, history, model_name, fold_idx, class_indices, metadata_used=False):
    """
    Log model and metrics to MLflow with proper signature and input example
    
    Args:
        model: The model to log
        history: Training history
        model_name: Name of the model
        fold_idx: Fold index for cross-validation
        class_indices: Dictionary mapping class names to indices
        metadata_used: Whether metadata was used in the model
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
            mlflow.log_param("metadata_used", metadata_used)  # Log whether metadata was used
            
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
            
            # Log the model with signature and input example
            try:
                # Save model to a temporary directory with proper .keras extension
                model_path = f"models/temp_model_{fold_idx}.keras"
                model.save(model_path)
                
                # Create an input example (dummy input with correct shape)
                if metadata_used and isinstance(model.input, list):
                    # For models with metadata, create appropriate input examples
                    image_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    metadata_dim = model.input[1].shape[1]
                    metadata_input = np.zeros((1, metadata_dim), dtype=np.float32)
                    input_example = [image_input, metadata_input]
                    
                    # Create model signature with the correct input format
                    output = model.predict(input_example)
                    signature = infer_signature(input_example, output)
                else:
                    # For standard image-only models
                    input_example = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    signature = infer_signature(input_example, model.predict(input_example))
                
                # Log the model to MLflow with signature and input example
                mlflow.tensorflow.log_model(
                    model, 
                    "model",
                    signature=signature,
                    input_example=input_example
                )
                
                # Clean up
                import shutil
                if os.path.exists(model_path):
                    os.remove(model_path)
            except Exception as e:
                print(f"Error logging model to MLflow: {e}")
    
    except Exception as e:
        print(f"Error in MLflow logging: {e}")
