from flask import Flask, request, jsonify
import numpy as np
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import torch
import sys
import traceback

# Add YOLOv5 directory to path
# YOLO_PATH = "./image_processing/yolov5"
YOLO_PATH = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/image_processing/yolov5"
sys.path.append(YOLO_PATH)

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = '/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/training_model/src/temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path configurations - Update these paths for your environment
script_dir = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/training_model/src/best.pt"
CLASSIFICATION_MODEL_PATH = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/training_model/src/skin_cancer_prediction_model.keras"
DETECTION_OUTPUT_DIR = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/training_model/src/annotated_images/annotated_images"

# Global variables for detection output directory
# This is used across functions to ensure consistency
def get_detection_output_dir():
    return DETECTION_OUTPUT_DIR

# Model loading
yolo_model = None
classification_model = None

# Load YOLOv5 model
def load_yolo_model():
    try:
        # Load YOLOv5 model
        model = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_MODEL_PATH, source='local')
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # IoU threshold
        # Set project name to None to avoid creating new numbered folders
        model.project = DETECTION_OUTPUT_DIR
        model.name = ''    # Empty string means don't create a subfolder
        model.exist_ok = True  # Overwrite existing files
        print("YOLOv5 model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        traceback.print_exc()
        return None

# Load classification model
def load_classification_model():
    try:
        # Load the trained model
        model = load_model(CLASSIFICATION_MODEL_PATH)
        print(f"Classification model loaded successfully from {CLASSIFICATION_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        traceback.print_exc()
        return None

# Function to preprocess image for classification
def preprocess_image_for_classification(image_path, bbox=None):
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Apply bounding box crop if provided
        if bbox is not None:
            width, height = img.size
            x_center, y_center, box_width, box_height = bbox
            
            # Convert normalized coordinates to pixel values
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)
            
            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Crop the image
            img = img.crop((x1, y1, x2, y2))
        
        # Resize to 224x224
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Create a batch of size 1
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        traceback.print_exc()
        return None

# Function to run YOLOv5 detection
def detect_skin_lesion(image_path, yolo_model):
    try:
        # Ensure the output directories exist
        os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
        
        # Get the image name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Run detection
        results = yolo_model(image_path)
        
        # Print detection results for debugging
        print(f"Detection results: {len(results.xyxy[0])} objects found")
        if len(results.xyxy[0]) > 0:
            print(f"First detection confidence: {results.xyxy[0][0][-1]}")
        
        # Save results - YOLOv5 might create a numbered directory (e.g., annotated_images1)
        # We'll capture the actual save path from the YOLOv5 output
        save_result = results.save(save_dir=DETECTION_OUTPUT_DIR)
        
        # Extract the actual save directory from output message (which is printed to console)
        actual_save_dir = None
        if hasattr(save_result, 'get') and save_result.get('save_dir'):
            actual_save_dir = save_result.get('save_dir')
            print(f"YOLOv5 saved results to: {actual_save_dir}")
        else:
            # Try to find the most recent numbered directory
            parent_dir = os.path.dirname(DETECTION_OUTPUT_DIR)
            base_name = os.path.basename(DETECTION_OUTPUT_DIR)
            potential_dirs = [d for d in os.listdir(parent_dir) 
                             if d.startswith(base_name) and os.path.isdir(os.path.join(parent_dir, d))]
            if potential_dirs:
                # Sort directories by modification time (most recent first)
                sorted_dirs = sorted(potential_dirs, 
                                    key=lambda d: os.path.getmtime(os.path.join(parent_dir, d)), 
                                    reverse=True)
                actual_save_dir = os.path.join(parent_dir, sorted_dirs[0])
                print(f"Found most recent save directory: {actual_save_dir}")
            else:
                actual_save_dir = DETECTION_OUTPUT_DIR
                print(f"Using default save directory: {actual_save_dir}")
                
        # Path to the label file - check both in the numbered directory and original directory
        label_path = os.path.join(actual_save_dir, 'labels', f"{image_name}.txt")
        if not os.path.exists(label_path):
            # Also check in original directory
            original_label_path = os.path.join(DETECTION_OUTPUT_DIR, 'labels', f"{image_name}.txt")
            if os.path.exists(original_label_path):
                label_path = original_label_path
                
        print(f"Looking for label file at: {label_path}")
        
        # Force a bbox if the model detected something but didn't save a label file
        if not os.path.exists(label_path) and len(results.xyxy[0]) > 0:
            print("Detection found but no label file exists, creating one")
            # Get normalized coordinates from the first detection
            box = results.xyxy[0][0].cpu().numpy()  # x1, y1, x2, y2, conf, class
            
            # Convert to YOLO format (class, x_center, y_center, width, height)
            img = Image.open(image_path)
            width, height = img.size
            
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            
            # Create label file manually
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center} {y_center} {box_width} {box_height} {box[4]}")
        
        # Check if label file exists (detection found)
        if not os.path.exists(label_path):
            print(f"No label file found at {label_path}, returning None")
            return None, None
        
        # Read the label file to get bounding box
        with open(label_path, 'r') as f:
            label_content = f.read().strip().split()
        
        print(f"Label content: {label_content}")
        
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
            detected_image_path = os.path.join(actual_save_dir, image_name + '.jpg')
            if not os.path.exists(detected_image_path):
                detected_image_path = os.path.join(actual_save_dir, image_name + '.png')
                
            # If still not found, check original directory
            if not os.path.exists(detected_image_path):
                alternate_path = os.path.join(DETECTION_OUTPUT_DIR, image_name + '.jpg')
                if os.path.exists(alternate_path):
                    detected_image_path = alternate_path
                else:
                    alternate_path = os.path.join(DETECTION_OUTPUT_DIR, image_name + '.png')
                    if os.path.exists(alternate_path):
                        detected_image_path = alternate_path
            
            # If we didn't find a saved image, use the original image
            if not os.path.exists(detected_image_path):
                print(f"No detected image found at {detected_image_path}, using original image")
                detected_image_path = image_path
            
            print(f"Detection successful, bbox={bbox}, image_path={detected_image_path}")
            return detected_image_path, bbox
        
        print("Label file exists but content is invalid")
        return None, None
    except Exception as e:
        print(f"Error in skin lesion detection: {e}")
        traceback.print_exc()
        return None, None

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ISIC dataset class mapping based on training
class_to_diagnosis = {
    0: "MEL",  # Melanoma
    1: "NV",   # Nevus
    2: "BCC",  # Basal Cell Carcinoma
    3: "AK",   # Actinic Keratosis
    4: "BKL",  # Benign Keratosis
    5: "DF",   # Dermatofibroma
    6: "VASC", # Vascular Lesion
    7: "SCC",  # Squamous Cell Carcinoma
    8: "UNK"   # Unknown
}

# Mapping to human-readable diagnosis names
diagnosis_to_full_name = {
    "MEL": "Melanoma",
    "NV": "Nevus (Mole)",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
    "UNK": "Unknown"
}

# Map diagnosis to benign/malignant status
diagnosis_to_status = {
    "MEL": "malignant",
    "NV": "benign",
    "BCC": "malignant",
    "AK": "pre-malignant",
    "BKL": "benign",
    "DF": "benign",
    "VASC": "benign",
    "SCC": "malignant",
    "UNK": "unknown"
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
            processed_img = preprocess_image_for_classification(detected_image_path, bbox)
            
            if processed_img is None:
                return jsonify({'error': 'Error processing the detected image'}), 500
            
            # Step 3: Make prediction with the classification model
            prediction = classification_model.predict(processed_img)
            
            # Get the predicted class index and probabilities
            if prediction.shape[1] > 1:
                # Multi-class prediction (using argmax)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class_idx])
                
                # Convert all class probabilities
                class_probabilities = {}
                for i, prob in enumerate(prediction[0]):
                    if i in class_to_diagnosis:
                        diagnosis_code = class_to_diagnosis[i]
                        class_probabilities[diagnosis_code] = {
                            'code': diagnosis_code,
                            'name': diagnosis_to_full_name.get(diagnosis_code, 'Unknown'),
                            'probability': float(prob),
                            'status': diagnosis_to_status.get(diagnosis_code, 'unknown')
                        }
            else:
                # Binary prediction
                predicted_class_idx = 1 if prediction[0][0] > 0.5 else 0
                confidence = float(prediction[0][0]) if predicted_class_idx == 1 else 1 - float(prediction[0][0])
                
                # Binary class probabilities
                class_probabilities = {
                    'benign': {
                        'code': 'BENIGN',
                        'name': 'Benign Lesion',
                        'probability': 1 - float(prediction[0][0]),
                        'status': 'benign'
                    },
                    'malignant': {
                        'code': 'MALIGNANT',
                        'name': 'Malignant Lesion',
                        'probability': float(prediction[0][0]),
                        'status': 'malignant'
                    }
                }
            # Get the diagnosis code
            diagnosis_code = class_to_diagnosis.get(predicted_class_idx, "UNK")
            
            # Get the full name and status
            diagnosis_name = diagnosis_to_full_name.get(diagnosis_code, "Unknown")
            status = diagnosis_to_status.get(diagnosis_code, "unknown")
            
            # Return the prediction result
            result = {
                'diagnosis_code': diagnosis_code,
                'diagnosis_name': diagnosis_name,
                'status': status,
                'confidence': confidence,
                'class_probabilities': class_probabilities
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
    return jsonify({
        'status': 'healthy', 
        'models_loaded': yolo_model is not None and classification_model is not None,
        'yolo_model': 'loaded' if yolo_model is not None else 'not loaded',
        'classification_model': 'loaded' if classification_model is not None else 'not loaded'
    })

# Load models when the app starts
print("Loading models...")
yolo_model = load_yolo_model()
classification_model = load_classification_model()

if __name__ == '__main__':
    # Check if models loaded successfully
    if yolo_model is None or classification_model is None:
        print("ERROR: Failed to load one or more models. Check the logs for details.")
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)