from flask import Flask, request, jsonify
import numpy as np
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Input
import torch
import sys
from PIL import Image

# Import custom layer with correct relative import
from custom_layers import CustomScaleLayer

# Import the preprocessing function from data_preprocessing
from src.data_preprocessing import preprocess_image as data_preprocess_image

# Add YOLOv5 directory to path
YOLO_PATH = "/Users/drake/Documents/UWE/DermaAI-Care/Training/image_processing/yolov5"
sys.path.append(YOLO_PATH)

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path configurations
YOLO_MODEL_PATH = "/Users/drake/Documents/UWE/DermaAI-Care/Training/image_processing/lesion_runs/yolov5x_skin_lesions4/weights/best.pt"
CLASSIFICATION_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/skin_cancer_prediction_model.keras')
DETECTION_OUTPUT_DIR = "/Users/drake/Documents/UWE/DermaAI-Care/Training/image_processing/yolov5/runs/detect/exp"

# Define custom objects dictionary for model loading
custom_objects = {
    'CustomScaleLayer': CustomScaleLayer
}

# Load YOLOv5 model
def load_yolo_model():
    try:
        # Load YOLOv5 model
        model = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_MODEL_PATH, source='local')
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # IoU threshold
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return None

# Create classification model architecture
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

# Function to run YOLOv5 detection
def detect_skin_lesion(image_path, yolo_model):
    # Run detection
    results = yolo_model(image_path)
    
    # Save results to the detection output directory
    results.save(save_dir=DETECTION_OUTPUT_DIR)
    
    # Get the image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Path to the label file
    label_path = os.path.join(DETECTION_OUTPUT_DIR, 'labels', f"{image_name}.txt")
    
    # Check if label file exists (detection found)
    if not os.path.exists(label_path):
        return None, None
    
    # Read the label file to get bounding box
    with open(label_path, 'r') as f:
        label_content = f.read().strip().split()
    
    # Parse label content (class x_center y_center width height confidence)
    if len(label_content) >= 5:
        class_id = int(label_content[0])
        x_center = float(label_content[1])
        y_center = float(label_content[2])
        width = float(label_content[3])
        height = float(label_content[4])
        confidence = float(label_content[5]) if len(label_content) > 5 else 1.0
        
        # Return the bounding box coordinates
        bbox = [x_center, y_center, width, height]
        
        # Path to the detected image
        detected_image_path = os.path.join(DETECTION_OUTPUT_DIR, image_name + '.jpg')
        if not os.path.exists(detected_image_path):
            detected_image_path = os.path.join(DETECTION_OUTPUT_DIR, image_name + '.png')
        
        return detected_image_path, bbox
    
    return None, None

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models at startup
print("Loading models...")
yolo_model = load_yolo_model()
classification_model = create_model_architecture()

# Try to load saved weights
try:
    classification_model.load_weights(CLASSIFICATION_MODEL_PATH)
    print(f"Classification model loaded successfully from {CLASSIFICATION_MODEL_PATH}")
except Exception as e:
    print(f"Error loading classification model: {e}")

# Map class indices to diagnosis names
class_to_diagnosis = {
    0: "nevus",
    1: "melanoma",
    2: "seborrheic keratosis",
    3: "basal cell carcinoma",
    4: "actinic keratosis",
    5: "squamous cell carcinoma"
}

# Map diagnosis to benign/malignant status
diagnosis_to_status = {
    "nevus": "benign",
    "melanoma": "malignant",
    "seborrheic keratosis": "benign",
    "basal cell carcinoma": "malignant",
    "actinic keratosis": "benign",
    "squamous cell carcinoma": "malignant"
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Step 1: Detect skin lesion using YOLOv5
            detected_image_path, bbox = detect_skin_lesion(file_path, yolo_model)
            
            if detected_image_path is None or bbox is None:
                return jsonify({'error': 'No skin lesion detected in the image'}), 400
            
            # Step 2: Preprocess the detected image with bounding box
            processed_img = data_preprocess_image(detected_image_path, bbox)
            
            # Step 3: Make prediction with the classification model
            prediction = classification_model.predict(processed_img)
            
            # Get the predicted class index
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            # Get the diagnosis and status
            diagnosis = class_to_diagnosis.get(predicted_class_idx, "unknown")
            status = diagnosis_to_status.get(diagnosis, "unknown")
            
            # Return the prediction result
            result = {
                'diagnosis': diagnosis,
                'status': status,
                'confidence': confidence,
                'class_probabilities': {class_to_diagnosis[i]: float(prob) for i, prob in enumerate(prediction[0])}
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({'error': 'Invalid file format. Allowed formats: png, jpg, jpeg'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': yolo_model is not None and classification_model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)