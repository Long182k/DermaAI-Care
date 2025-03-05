from sklearn.model_selection import KFold
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Define strategy at module level so it can be imported
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

def build_model(num_classes):
    """
    Build and compile the model with memory optimizations
    """
    with strategy.scope():
        # Load the pre-trained InceptionResNetV2 model with smaller input size
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'  # Use average pooling to handle dimensions consistently
        )
        
        # Freeze all layers initially
        base_model.trainable = False
        
        # Build the model using Functional API with proper dimension handling
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        # Remove the separate GlobalAveragePooling2D since we use pooling='avg' in base_model
        x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use memory-efficient optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model

def cross_validate_model(model, data, labels, k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    scores = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data)):
        # Train and evaluate model for each fold
        history = model.fit(
            data[train_ids], labels[train_ids],
            validation_data=(data[val_ids], labels[val_ids]),
            epochs=10
        )
        scores.append(history.history['val_accuracy'][-1]) 

def train_model(model, train_generator, val_generator, epochs, class_weights=None, early_stopping_patience=3, reduce_lr_patience=2):
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
            monitor='val_accuracy',
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir='./backup'),
        tf.keras.callbacks.TerminateOnNaN()
    ]
    
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", train_generator.batch_size)
        
        if hasattr(model.optimizer, 'learning_rate'):
            if hasattr(model.optimizer.learning_rate, 'initial_learning_rate'):
                initial_lr = model.optimizer.learning_rate.initial_learning_rate
            else:
                initial_lr = float(model.optimizer.learning_rate)
        else:
            initial_lr = float(model.optimizer._learning_rate)
            
        mlflow.log_param("initial_learning_rate", initial_lr)
        
        # Use mixed precision for faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Configure training to be thread-safe and optimize for performance
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        options.experimental_optimization.parallel_batch = True
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Log metrics
        mlflow.log_metrics({
            "final_accuracy": history.history['accuracy'][-1],
            "final_val_accuracy": history.history['val_accuracy'][-1],
            "final_auc": history.history['auc'][-1],
            "final_val_auc": history.history['val_auc'][-1],
            "final_precision": history.history['precision'][-1],
            "final_val_precision": history.history['val_precision'][-1],
            "final_recall": history.history['recall'][-1],
            "final_val_recall": history.history['val_recall'][-1]
        })
        
    return history

def fine_tune_model(model, train_generator, val_generator, epochs=1, early_stopping_patience=2):
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
            
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=True)
            x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(model.output_shape[-1], activation='softmax', kernel_initializer='he_normal')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Restore weights
            try:
                model.set_weights(weights)
                print("Weights restored successfully")
            except ValueError as e:
                print(f"Error restoring weights: {e}")
                print("Creating fresh model for fine-tuning...")
            
            # Unfreeze the last few layers of the base model
            base_model.trainable = True
            for layer in base_model.layers[:-20]:  # Freeze all except last 20 layers
                layer.trainable = False
            
            # Recompile with a lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy',
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')]
            )
    
    # Train with fine-tuning
    history = train_model(
        model,
        train_generator,
        val_generator,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=1  # Faster LR reduction during fine-tuning
    )
    
    return history