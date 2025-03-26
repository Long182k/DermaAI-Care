import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
import glob

def preprocess_image(image_path, bbox=None):
    """
    Preprocess single image according to InceptionResNetV2 requirements
    If bbox is provided, crop the image to the bounding box first
    """
    # Load image
    img = Image.open(image_path)
    
    # If bounding box is provided, crop the image
    if bbox is not None:
        width, height = img.size
        x_center, y_center, box_width, box_height = bbox
        
        # Convert normalized coordinates to pixel values
        x1 = int((x_center - box_width/2) * width)
        y1 = int((y_center - box_height/2) * height)
        x2 = int((x_center + box_width/2) * width)
        y2 = int((y_center + box_height/2) * height)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        # Crop the image
        img = img.crop((x1, y1, x2, y2))
    
    # Resize to target size
    img = img.resize((224, 224))
    
    # Convert to array
    img_array = img_to_array(img)
    
    # Expand dimensions for batch processing
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply InceptionResNetV2 preprocessing
    processed_img = preprocess_input(img_array)
    
    return processed_img

def create_augmentation_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

def load_yolo_detections(image_dir, labels_dir, csv_path, min_samples_per_class=2, n_folds=5):
    """
    Load YOLOv5 detected images and their corresponding labels
    """
    # Read the CSV file for diagnosis information
    df = pd.read_csv(csv_path)
    
    # Get all PNG images from the detection directory
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    
    # Create a new dataframe for detected images
    detected_data = []
    
    for img_path in image_files:
        # Extract image name from path
        img_name = os.path.basename(img_path).split('.')[0]
        
        # Find corresponding label file
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            continue
            
        # Find the original image entry in the CSV
        original_entry = df[df['image_name'] == img_name]
        
        # Skip if no matching entry in CSV
        if len(original_entry) == 0:
            continue
            
        # Read the label file to get bounding box
        with open(label_path, 'r') as f:
            label_content = f.read().strip().split()
            
        # Parse label content (class x_center y_center width height confidence)
        if len(label_content) >= 5:  # Ensure we have at least the basic bbox info
            class_id = int(label_content[0])
            x_center = float(label_content[1])
            y_center = float(label_content[2])
            width = float(label_content[3])
            height = float(label_content[4])
            confidence = float(label_content[5]) if len(label_content) > 5 else 1.0
            
            # Add to detected data
            detected_data.append({
                'image_name': img_name,
                'image_path': img_path,
                'diagnosis': original_entry['diagnosis'].values[0],
                'benign_malignant': original_entry['benign_malignant'].values[0],
                'target': original_entry['target'].values[0],
                'bbox': [x_center, y_center, width, height],
                'confidence': confidence
            })
    
    # Create dataframe from detected data
    detected_df = pd.DataFrame(detected_data)
    
    # Count samples per class
    class_counts = detected_df['diagnosis'].value_counts()
    print("\nSamples per class before filtering:")
    print(class_counts)
    
    # Filter out classes with too few samples
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    detected_df = detected_df[detected_df['diagnosis'].isin(valid_classes)]
    
    print("\nSamples per class after filtering:")
    print(detected_df['diagnosis'].value_counts())
    
    # Create class mappings for diagnosis
    diagnosis_to_idx = {diagnosis: idx for idx, diagnosis in enumerate(detected_df['diagnosis'].unique())}
    
    print(f"\nNumber of classes: {len(diagnosis_to_idx)}")
    print("Classes:", list(diagnosis_to_idx.keys()))
    
    try:
        # Initialize KFold cross validator
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store fold indices
        fold_indices = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(detected_df)):
            fold_indices.append({
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
            print(f"\nFold {fold_idx + 1}:")
            print(f"Training size: {len(train_idx)}")
            print(f"Validation size: {len(val_idx)}")
        
        return detected_df, fold_indices, diagnosis_to_idx
        
    except Exception as e:
        print(f"\nError in k-fold split: {str(e)}")
        print("Dataset statistics:")
        print(detected_df['diagnosis'].value_counts())
        raise

class YOLODetectionGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, diagnosis_to_idx, batch_size=64, is_training=True):
        super().__init__()
        self.df = dataframe
        self.batch_size = batch_size
        self.diagnosis_to_idx = diagnosis_to_idx
        self.is_training = is_training
        self.augmentation = create_augmentation_pipeline() if is_training else None
        self.n_classes = len(diagnosis_to_idx)
        self.indices = np.arange(len(self.df))
        
        # Pre-compute image paths to avoid memory leaks
        self.image_paths = self.df['image_path'].values
        self.diagnoses = self.df['diagnosis'].values
        self.bboxes = self.df['bbox'].values
        
        if is_training:
            np.random.shuffle(self.indices)
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = np.zeros((len(batch_indices), 224, 224, 3))
        batch_y = np.zeros((len(batch_indices), self.n_classes))
        
        for i, idx in enumerate(batch_indices):
            # Load image and crop to bounding box
            img = Image.open(self.image_paths[idx])
            
            # Apply bounding box cropping
            bbox = self.bboxes[idx]
            width, height = img.size
            x_center, y_center, box_width, box_height = bbox
            
            # Convert normalized coordinates to pixel values
            x1 = int((x_center - box_width/2) * width)
            y1 = int((y_center - box_height/2) * height)
            x2 = int((x_center + box_width/2) * width)
            y2 = int((y_center + box_height/2) * height)
            
            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Crop the image
            img = img.crop((x1, y1, x2, y2))
            
            # Resize to target size
            img = img.resize((224, 224))
            img_array = img_to_array(img)
            
            if self.is_training and self.augmentation:
                img_array = self.augmentation(img_array)
            
            img_array = preprocess_input(img_array)
            
            batch_x[i] = img_array
            batch_y[i] = tf.keras.utils.to_categorical(
                self.diagnosis_to_idx[self.diagnoses[idx]], 
                self.n_classes
            )
        
        return batch_x, batch_y

# In the create_yolo_generators function
def create_yolo_generators(csv_path, image_dir, labels_dir, batch_size, min_samples_per_class, n_folds, fold_idx):
    """
    Create train and validation generators for a specific fold using YOLO detected images
    """
    df, fold_indices, diagnosis_to_idx = load_yolo_detections(
        image_dir, 
        labels_dir,
        csv_path,
        min_samples_per_class=min_samples_per_class,
        n_folds=n_folds
    )
    
    # Get indices for the specified fold
    fold_data = fold_indices[fold_idx]
    train_idx = fold_data['train_idx']
    val_idx = fold_data['val_idx']
    
    # Create train and validation dataframes for this fold
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # Create generators with smaller batch size
    train_generator = YOLODetectionGenerator(
        train_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=True
    )
    
    val_generator = YOLODetectionGenerator(
        val_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=False
    )
    
    n_classes = len(diagnosis_to_idx)
    # Create generators with prefetching for better GPU utilization
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(diagnosis_to_idx)), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(diagnosis_to_idx)), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, diagnosis_to_idx, n_classes, len(fold_indices)

def analyze_yolo_dataset(csv_path, image_dir, labels_dir):
    """
    Analyze dataset statistics for YOLO detected images
    """
    # Read the CSV file for diagnosis information
    df = pd.read_csv(csv_path)
    
    # Get all PNG images from the detection directory
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    
    # Create a new dataframe for detected images
    detected_data = []
    
    for img_path in image_files:
        # Extract image name from path
        img_name = os.path.basename(img_path).split('.')[0]
        
        # Find corresponding label file
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            continue
            
        # Find the original image entry in the CSV
        original_entry = df[df['image_name'] == img_name]
        
        # Skip if no matching entry in CSV
        if len(original_entry) == 0:
            continue
            
        # Read the label file to get bounding box
        with open(label_path, 'r') as f:
            label_content = f.read().strip().split()
            
        # Parse label content (class x_center y_center width height confidence)
        if len(label_content) >= 5:  # Ensure we have at least the basic bbox info
            # Add to detected data
            detected_data.append({
                'image_name': img_name,
                'diagnosis': original_entry['diagnosis'].values[0],
                'benign_malignant': original_entry['benign_malignant'].values[0],
                'target': original_entry['target'].values[0]
            })
    
    # Create dataframe from detected data
    detected_df = pd.DataFrame(detected_data)
    
    print("Dataset Statistics:")
    print("-" * 50)
    print(f"Total number of detected images: {len(detected_df)}")
    print("\nDiagnosis distribution:")
    diagnosis_counts = detected_df['diagnosis'].value_counts()
    print(diagnosis_counts)
    
    print("\nClasses with less than 2 samples:")
    print(diagnosis_counts[diagnosis_counts < 2])
    
    print("\nBenign/Malignant distribution:")
    print(detected_df['benign_malignant'].value_counts())
    
    # Calculate class weights for imbalanced data (only for classes with enough samples)
    valid_classes = diagnosis_counts[diagnosis_counts >= 2].index
    valid_df = detected_df[detected_df['diagnosis'].isin(valid_classes)]
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(valid_df['diagnosis']),
        y=valid_df['diagnosis']
    )
    
    return {
        'class_weights': dict(zip(np.unique(valid_df['diagnosis']), class_weights)),
        'n_classes': len(valid_classes)
    }

# Usage example:
if __name__ == "__main__":
    CSV_PATH = "/Users/drake/Documents/UWE/DermaAI-Care/Training/data/ISIC_2020_Training_GroundTruth_v2.csv"
    IMAGE_DIR = "/Users/drake/Documents/UWE/DermaAI-Care/Training/image_processing/yolov5/runs/detect/exp"
    
    # Analyze dataset
    dataset_stats = analyze_dataset(CSV_PATH)
    print("\nClass weights for handling imbalance:")
    print(dataset_stats['class_weights'])
    
    # Create generators
    train_gen, val_gen, diagnosis_to_idx, n_classes, fold_size = create_generators(
        CSV_PATH,
        IMAGE_DIR,
        fold_idx=0
    )