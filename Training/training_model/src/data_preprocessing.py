import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import gc
import random

# Set seeds for reproducibility
def set_random_seeds(seed=42):
    """Set random seeds for TensorFlow, NumPy and Python"""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set determinism configuration
    if hasattr(tf.config, 'experimental'):
        tf.config.experimental.enable_op_determinism()

# Call this function at the beginning
set_random_seeds()

def create_augmentation_pipeline(seed=42):
    """
    Create an augmentation pipeline that's memory-efficient and deterministic
    """
    # These operations work well on GPU with deterministic settings
    gpu_augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2, seed=seed),
        tf.keras.layers.RandomZoom(0.2, seed=seed),
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomBrightness(0.2, seed=seed),
    ])
    
    return gpu_augmentations

class YOLODetectionGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, diagnosis_to_idx, batch_size=16, is_training=True, seed=42, memory_efficient=True):
        super().__init__()
        self.df = dataframe
        self.batch_size = batch_size  # Reduced batch size
        self.diagnosis_to_idx = diagnosis_to_idx
        self.is_training = is_training
        self.seed = seed
        self.memory_efficient = memory_efficient
        self.augmentation = create_augmentation_pipeline(seed) if is_training else None
        self.n_classes = len(diagnosis_to_idx)
        self.indices = np.arange(len(self.df))
        
        # Pre-compute image paths to avoid memory leaks
        self.image_paths = self.df['image_path'].values
        self.diagnoses = self.df['diagnosis'].values
        self.bboxes = self.df['bbox'].values
        
        # Add class indices mapping
        self.class_indices = {v: k for k, v in diagnosis_to_idx.items()}
        
        if is_training:
            # Use the seed for reproducible shuffling
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self.indices)
    
    def on_epoch_end(self):
        if self.is_training:
            # Use a different seed for each epoch but still deterministic
            epoch_seed = self.seed + self.current_epoch if hasattr(self, 'current_epoch') else self.seed
            rng = np.random.RandomState(epoch_seed)
            rng.shuffle(self.indices)
            if hasattr(self, 'current_epoch'):
                self.current_epoch += 1
            else:
                self.current_epoch = 1
    
    def __len__(self):
        # Ensure all batches have the same size by dropping the last incomplete batch
        return max(1, len(self.df) // self.batch_size)  # Ensure at least 1 batch
    
    def __getitem__(self, idx):
        # Get indices for this batch
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        # If we don't have enough samples, repeat some to fill the batch
        if len(batch_indices) < self.batch_size:
            # Use deterministic random choice with seed
            rng = np.random.RandomState(self.seed + idx)  # Different seed for each batch but deterministic
            extra_needed = self.batch_size - len(batch_indices)
            extra_indices = rng.choice(batch_indices, size=extra_needed, replace=True)
            batch_indices = np.concatenate([batch_indices, extra_indices])
        
        # Now we're guaranteed to have exactly batch_size samples
        batch_x = np.zeros((self.batch_size, 224, 224, 3), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, self.n_classes), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            try:
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
                
                # Memory optimization: explicitly delete PIL image
                del img
                
                if self.is_training and self.augmentation:
                    # Apply GPU-friendly augmentations
                    img_array = self.augmentation(np.expand_dims(img_array, axis=0))[0]
                    
                    # Apply contrast using NumPy instead of TensorFlow
                    item_seed = self.seed + idx + i  # Unique but deterministic seed for each item
                    if item_seed % 2 == 0:  # Apply to ~50% of images
                        # Set the random seed for NumPy
                        np.random.seed(item_seed)
                        
                        # Apply contrast adjustment using NumPy (CPU operation)
                        contrast_factor = np.random.uniform(0.8, 1.2)
                        mean = np.mean(img_array, axis=(0, 1), keepdims=True)
                        img_array = (img_array - mean) * contrast_factor + mean
                        img_array = np.clip(img_array, 0, 255)
                
                # Apply preprocessing
                img_array = preprocess_input(img_array)
                
                batch_x[i] = img_array
                batch_y[i] = tf.keras.utils.to_categorical(
                    self.diagnosis_to_idx[self.diagnoses[idx]], 
                    self.n_classes
                )
                
                # Memory optimization: explicitly delete arrays
                if self.memory_efficient:
                    del img_array
                    # Force garbage collection periodically
                    if i % 8 == 0:
                        gc.collect()
            except Exception as e:
                print(f"Error processing image {self.image_paths[idx]}: {e}")
                # Use a blank image if there's an error
                batch_x[i] = np.zeros((224, 224, 3), dtype=np.float32)
                batch_y[i] = tf.keras.utils.to_categorical(
                    self.diagnosis_to_idx[self.diagnoses[idx]], 
                    self.n_classes
                )
        
        return batch_x, batch_y

def load_yolo_detections(image_dir, labels_dir, csv_path, min_samples_per_class=5, n_folds=5):
    """
    Load YOLO detected images and create cross-validation folds
    """
    # Read the CSV file for diagnosis information
    df = pd.read_csv(csv_path)
    
    # Get all JPG images from the detection directory
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
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
            # Extract bounding box coordinates
            bbox = [float(label_content[1]), float(label_content[2]), 
                   float(label_content[3]), float(label_content[4])]
            
            # Add to detected data
            detected_data.append({
                'image_name': img_name,
                'image_path': img_path,
                'diagnosis': original_entry['diagnosis'].values[0],
                'benign_malignant': original_entry['benign_malignant'].values[0],
                'target': original_entry['target'].values[0],
                'bbox': bbox
            })
    
    # Create dataframe from detected data
    detected_df = pd.DataFrame(detected_data)
    
    # Filter out classes with too few samples
    class_counts = detected_df['diagnosis'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    filtered_df = detected_df[detected_df['diagnosis'].isin(valid_classes)]
    
    # Create diagnosis to index mapping
    unique_diagnoses = filtered_df['diagnosis'].unique()
    diagnosis_to_idx = {diagnosis: i for i, diagnosis in enumerate(unique_diagnoses)}
    
    # Create stratified k-fold indices
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_indices = []
    
    for train_idx, val_idx in skf.split(filtered_df, filtered_df['diagnosis']):
        fold_indices.append({
            'train_idx': train_idx,
            'val_idx': val_idx
        })
    
    return filtered_df, fold_indices, diagnosis_to_idx

def create_yolo_generators(csv_path, image_dir, labels_dir, batch_size=16, min_samples_per_class=5, n_folds=5, fold_idx=0, seed=42):
    """
    Create train and validation generators for a specific fold using YOLO detected images
    Uses a smaller batch size by default (16) to reduce memory usage
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
    
    # Create generators with memory efficiency enabled
    train_generator = YOLODetectionGenerator(
        train_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=True,
        seed=seed,
        memory_efficient=True
    )
    
    val_generator = YOLODetectionGenerator(
        val_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=False,
        seed=seed+1,
        memory_efficient=True
    )
    
    # Access class indices before converting to dataset
    train_class_indices = train_generator.class_indices
    val_class_indices = val_generator.class_indices
    
    n_classes = len(diagnosis_to_idx)
    
    return train_generator, val_generator, diagnosis_to_idx, n_classes, len(fold_indices), train_class_indices, val_class_indices

def analyze_yolo_dataset(csv_path, image_dir, labels_dir):
    """
    Analyze dataset statistics for YOLO detected images
    """
    # Read the CSV file for diagnosis information
    df = pd.read_csv(csv_path)
    
    # Get all PNG images from the detection directory
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
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
    
    print("class_weights data_preprocessing", class_weights)
    print("dict(zip(np.unique(valid_df['diagnosis']), class_weights))", dict(zip(np.unique(valid_df['diagnosis']), class_weights)))
   
    return {
        'class_weights': dict(zip(np.unique(valid_df['diagnosis']), class_weights)),
        'n_classes': len(valid_classes),
        'dataset_size': len(detected_df),  # Total number of images
        'valid_dataset_size': len(valid_df)  # Number of images in valid classes
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