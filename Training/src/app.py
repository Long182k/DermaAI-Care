from flask import Flask, request, jsonify
import numpy as np
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Input

# Import custom layer with correct relative import
from custom_layers import CustomScaleLayer

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/skinning_cancer_prediction_model.h5')

# Define custom objects dictionary for model loading
custom_objects = {
    'CustomScaleLayer': CustomScaleLayer
}

def create_model_architecture(num_classes=6):  # Adjust num_classes to match saved weights
    # Clear any existing models from memory
    tf.keras.backend.clear_session()
    
    # Create the base model with the same architecture as in training
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
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with the same metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

# Load the model with custom objects
try:
    # Set memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Recreate the model architecture with the correct number of classes
    model = create_model_architecture(num_classes=6)  # Ensure this matches the saved model
    
    # Load weights from the saved model
    model.load_weights(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    print("WARNING: Model not loaded. Predictions will not work!")
    print("Make sure to train the model first using train.py")

def preprocess_image(image_path):
    """
    Preprocess single image according to InceptionResNetV2 requirements
    """
    # Load image in target size (224x224)
    img = load_img(image_path, target_size=(224, 224))
    
    # Convert to array
    img_array = img_to_array(img)
    
    # Expand dimensions for batch processing
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply InceptionResNetV2 preprocessing
    processed_img = preprocess_input(img_array)
    
    return processed_img

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
            
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_img = preprocess_image(filepath)
        
        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the first element
        confidence = float(predictions[0][predicted_class])
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'class': int(predicted_class),
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Run the app
    print("Starting DermaAI-Care prediction service...")
    print(f"Model status: {'Loaded' if model is not None else 'Not loaded'}")
    app.run(host='0.0.0.0', port=5000, debug=False)  # Set debug=False in production