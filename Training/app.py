import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import cv2
import timm
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import base64
from ultralytics import YOLO
import time

app = Flask(__name__)

# Define paths to models
DETECTION_MODEL_PATH = '/Users/drake/Documents/UWE/IT_PROJECT/Code/DermaAI-Care/Training/best-v11.pt'
CLASSIFICATION_MODEL_PATH = '/Users/drake/Documents/UWE/IT_PROJECT/Code/DermaAI-Care/Training/the_best_model_2.pth'

# Class mapping for classification
CLASS_MAPPING = {0: 'AK', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'SCC', 7: 'VASC'}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLO model for detection
print("Loading detection model...")
try:
    detection_model = YOLO(DETECTION_MODEL_PATH)
    detection_model.to(device)
    print("Detection model loaded successfully")
except Exception as e:
    print(f"Error loading detection model: {e}")
    raise

# Define the classification model architecture
class LesionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(LesionClassifier, self).__init__()
        self.model = timm.create_model('seresnext101_32x4d', pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Apply LoRA
        target_modules = [
            name for name, module in self.model.named_modules()
            if isinstance(module, nn.Conv2d) and ('layer3' in name or 'layer4' in name)
        ]
        if not target_modules:
            raise ValueError("No Conv2d modules found in layer3 or layer4.")
        
        config = LoraConfig(
            target_modules=target_modules,
            r=16,
            lora_alpha=32,
            lora_dropout=0.2,
        )
        self.model = get_peft_model(self.model, config)
    
    def forward(self, x):
        return self.model(x)

# Load the classification model
print("Loading classification model...")
try:
    classification_model = LesionClassifier(num_classes=8)
    classification_model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=device), strict=False)
    classification_model.to(device)
    classification_model.eval()
    print("Classification model loaded successfully with strict=False")
except Exception as e:
    print(f"Error loading classification model: {e}")
    raise

# Image preprocessing for classification
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Temperature for scaling (experiment with values > 1 to soften probabilities)
TEMPERATURE = 2.0

@app.route('/predict', methods=['POST'])
def detect_and_classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get the image from the request
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    print("==========img=============", img)
    
    # Step 1: Detect lesions using YOLO
    results = detection_model(img)
    result = results[0]
    detections = result.to_df().to_dict(orient="records")
    print("Detections:", detections)
    
    if not detections:
        return jsonify({
            'message': 'No lesions detected in the image',
            'detections': [],
            'image_with_boxes': None,
            'image_name': file.filename
        }), 200
    
    # Convert PIL image to NumPy array
    img_np = np.array(img)
    img_with_boxes = img_np.copy()
    
    # Directory to save results
    save_dir = 'results/detect'
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    timestamp = int(time.time())  # Unique timestamp
    
    results_with_classification = []
    
    # Process each detected lesion
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = int(detection['box']['x1']), int(detection['box']['y1']), int(detection['box']['x2']), int(detection['box']['y2'])
        
        # Crop the lesion
        cropped_lesion = img_np[y1:y2, x1:x2]
        if cropped_lesion.size == 0:
            print("Warning: Cropped lesion is empty. Skipping classification.")
            continue
        
        # Draw bounding box on the image
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save cropped lesion image
        cropped_filename = f'cropped_lesion_{idx}_{timestamp}.jpg'
        cv2.imwrite(os.path.join(save_dir, cropped_filename), cv2.cvtColor(cropped_lesion, cv2.COLOR_RGB2BGR))
        
        # Convert cropped lesion to PIL for classification
        cropped_pil = Image.fromarray(cropped_lesion)
        input_tensor = preprocess(cropped_pil).unsqueeze(0).to(device)
        
        # Classify the lesion
        with torch.no_grad():
            output = classification_model(input_tensor)
            # Apply temperature scaling
            scaled_output = output / TEMPERATURE
            probabilities = nn.functional.softmax(scaled_output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        # Log intermediate outputs for debugging
        print(f"==========output (logits)=========== {output}")
        print(f"==========scaled_output=========== {scaled_output}")
        print(f"==========probabilities=========== {probabilities}")
        print(f"==========predicted_class_idx=========== {predicted_class_idx}")
        print(f"==========confidence=========== {confidence}")
        
        # Compile results
        detection_result = {
            'bbox': [x1, y1, x2, y2],
            'detection_confidence': float(detection['confidence']),
            'class': CLASS_MAPPING[predicted_class_idx],
            'class_confidence': float(confidence)
        }
        
        # Add label to the image
        # label = f"{CLASS_MAPPING[predicted_class_idx]}: {confidence:.2f}"
        # label = f"{'NV'}: {float(detection['confidence'])}"
        # cv2.putText(img_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        results_with_classification.append(detection_result)
    
        # Save image with bounding boxes
        boxes_filename = f'image_with_boxes_{timestamp}.jpg'
        img_with_boxes_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, boxes_filename), img_with_boxes_bgr)

        # Encode the image with bounding boxes as base64
        _, buffer = cv2.imencode('.jpg', img_with_boxes_bgr)
        img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'message': f'Found {len(results_with_classification)} lesion(s)',
        'detections': results_with_classification,
        'image_with_boxes': img_str,
        'image_name': file.filename,  # Add the original image name
        'processed_image_name': boxes_filename  # Add the processed image filename
    }), 200

@app.route('/classify', methods=['POST'])
def classify_only():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = classification_model(input_tensor)
        # Apply temperature scaling
        scaled_output = output / TEMPERATURE
        probabilities = nn.functional.softmax(scaled_output[0], dim=0)
        top3_values, top3_indices = torch.topk(probabilities, 3)
        
        results = [
            {'class': CLASS_MAPPING[idx.item()], 'confidence': float(val.item())}
            for val, idx in zip(top3_values, top3_indices)
        ]
    
    return jsonify({
        'message': 'Classification successful',
        'top_predictions': results
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'detection_model': DETECTION_MODEL_PATH,
        'classification_model': CLASSIFICATION_MODEL_PATH
    }), 200

# Additional Debugging Tips for Low Class Confidence:
# 1. Visually inspect saved cropped images in 'results/detect' to ensure they contain the full lesion.
# 2. Test classification model on known training images to verify confidence scores.
#    - Load a few training images, apply the same preprocessing, and check the model's confidence.
#    - If confidence is high on training data, the issue might be with the test image or preprocessing mismatch.
#    - If confidence is low on training data, consider retraining the model with normalization or more data.
# 3. If confidence remains low, consider retraining the model with more balanced data.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)