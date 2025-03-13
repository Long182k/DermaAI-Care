from sklearn.model_selection import KFold
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

def build_model(num_classes):
    """
    Build and compile the model with memory optimizations
    """
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    with strategy.scope():
        # Load the pre-trained InceptionResNetV2 model with smaller input size
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze all layers initially
        base_model.trainable = False
        
        # Simplified classification head
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
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
        
        # Build the model
        model.build((None, 224, 224, 3))
        
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

def train_model(model, train_generator, val_generator, epochs=10, class_weights=None):
    """
    Enhanced training function with proper distribution strategy
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode='max'
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
    
    # Remove the distribution strategy from here
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
        
        # Configure training to be thread-safe
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        
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

def fine_tune_model(model, train_generator, val_generator, epochs=5):
    """
    Fine-tune the model by unfreezing some layers
    """
    # Unfreeze the last few layers of the base model
    base_model = model.layers[1]  # InceptionResNetV2 is the second layer
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with frozen layers
    history = train_model(
        model,
        train_generator,
        val_generator,
        epochs=epochs
    )
    
    return history 