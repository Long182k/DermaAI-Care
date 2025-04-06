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
    if hasattr(tf.config, "experimental"):
        tf.config.experimental.enable_op_determinism()


# Call this function at the beginning
set_random_seeds()


# Replace the create_augmentation_pipeline function
def create_augmentation_pipeline(seed=42, strength='medium'):
    """
    Create an augmentation pipeline with configurable strength
    """
    # Base augmentations for all strength levels
    base_augmentations = [
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.2, seed=seed),
    ]
    
    # Add more augmentations based on strength
    if strength == 'medium' or strength == 'high':
        base_augmentations.extend([
            tf.keras.layers.RandomZoom(0.2, seed=seed),
            tf.keras.layers.RandomBrightness(0.2, seed=seed),
        ])
    
    if strength == 'high':
        base_augmentations.extend([
            tf.keras.layers.RandomContrast(0.2, seed=seed),
            # More aggressive rotation
            tf.keras.layers.RandomRotation(0.3, seed=seed),
            # Color jitter
            tf.keras.layers.RandomTranslation(0.1, 0.1, seed=seed),
        ])
    
    return tf.keras.Sequential(base_augmentations)


# Add the image_shape attribute to the YOLODetectionGenerator class
class YOLODetectionGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        diagnosis_to_idx,
        batch_size=16,
        is_training=True,
        seed=42,
        memory_efficient=True,
        augmentation_strength='medium'
    ):
        super().__init__()
        self.df = dataframe
        self.batch_size = batch_size
        self.diagnosis_to_idx = diagnosis_to_idx
        self.is_training = is_training
        self.seed = seed
        self.memory_efficient = memory_efficient
        self.augmentation = create_augmentation_pipeline(seed, augmentation_strength) if is_training else None
        self.n_classes = len(diagnosis_to_idx)
        self.indices = np.arange(len(self.df))
        self.image_shape = (224, 224, 3)
        
        # Pre-compute image paths to avoid memory leaks
        self.image_paths = self.df["image_path"].values
        self.diagnoses = self.df["diagnosis"].values
        self.bboxes = self.df["bbox"].values

        # Add class indices mapping
        self.class_indices = {v: k for k, v in diagnosis_to_idx.items()}

        if is_training:
            # Use the seed for reproducible shuffling
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self.indices)

    def on_epoch_end(self):
        if self.is_training:
            # Use a different seed for each epoch but still deterministic
            epoch_seed = (
                self.seed + self.current_epoch
                if hasattr(self, "current_epoch")
                else self.seed
            )
            rng = np.random.RandomState(epoch_seed)
            rng.shuffle(self.indices)
            if hasattr(self, "current_epoch"):
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
            rng = np.random.RandomState(
                self.seed + idx
            )  # Different seed for each batch but deterministic
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
                x1 = int((x_center - box_width / 2) * width)
                y1 = int((y_center - box_height / 2) * height)
                x2 = int((x_center + box_width / 2) * width)
                y2 = int((y_center + box_height / 2) * height)

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
                    item_seed = (
                        self.seed + idx + i
                    )  # Unique but deterministic seed for each item
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
                    self.diagnosis_to_idx[self.diagnoses[idx]], self.n_classes
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
                    self.diagnosis_to_idx[self.diagnoses[idx]], self.n_classes
                )

        return batch_x, batch_y


# Add this function to load and process metadata
def load_metadata(metadata_csv_path):
    """
    Load and process metadata from ISIC dataset
    
    Args:
        metadata_csv_path: Path to the metadata CSV file
        
    Returns:
        Processed metadata DataFrame with encoded categorical features
    """
    # Load metadata
    metadata_df = pd.read_csv(metadata_csv_path)
    
    # Fill missing values
    metadata_df['age_approx'] = metadata_df['age_approx'].fillna(metadata_df['age_approx'].median())
    metadata_df['anatom_site_general'] = metadata_df['anatom_site_general'].fillna('unknown')
    metadata_df['sex'] = metadata_df['sex'].fillna('unknown')
    
    # Encode categorical features
    # One-hot encode anatomical site
    site_dummies = pd.get_dummies(metadata_df['anatom_site_general'], prefix='site')
    
    # One-hot encode sex
    sex_dummies = pd.get_dummies(metadata_df['sex'], prefix='sex')
    
    # Normalize age to 0-1 range
    metadata_df['age_normalized'] = metadata_df['age_approx'] / 100.0
    
    # Combine all features
    processed_metadata = pd.concat([
        metadata_df[['image']], 
        metadata_df[['age_normalized']], 
        site_dummies, 
        sex_dummies
    ], axis=1)
    
    print(f"Processed metadata with {len(processed_metadata)} entries and {processed_metadata.shape[1]} features")
    return processed_metadata

# Update the load_yolo_detections function to incorporate metadata
def load_yolo_detections(image_dir, labels_dir, csv_path, metadata_csv_path, min_samples_per_class=5, n_folds=5):
    """
    Load YOLO detected images with their corresponding labels, ground truth, and metadata
    
    Args:
        image_dir: Directory containing the images
        labels_dir: Directory containing YOLO labels
        csv_path: Path to the ground truth CSV
        metadata_csv_path: Path to the metadata CSV
        min_samples_per_class: Minimum samples required for a class to be included
        n_folds: Number of folds for cross-validation
        
    Returns:
        DataFrame with image paths, labels, metadata, fold indices, and class mapping
    """
    # Load ground truth data
    gt_df = pd.read_csv(csv_path)
    
    # Load and process metadata
    metadata_df = load_metadata(metadata_csv_path)
    
    # Get all image files from the directory
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.jpeg"))
    image_names = [os.path.basename(f).split('.')[0] for f in image_files]
    
    # Create a dataframe with image paths
    df = pd.DataFrame({
        'image_name': image_names,
        'image_path': image_files
    })
    
    # Add label paths
    df['label_path'] = df['image_name'].apply(lambda x: os.path.join(labels_dir, f"{x}.txt"))
    
    # Filter to only include images that have YOLO labels
    df = df[df['label_path'].apply(os.path.exists)]
    
    # Merge with ground truth data
    df['image'] = df['image_name']  # Create column to match with ground truth
    df = pd.merge(df, gt_df, on='image', how='inner')
    
    # Merge with metadata
    df = pd.merge(df, metadata_df, on='image', how='left')
    
    # Fill missing metadata values
    metadata_columns = [col for col in df.columns if col.startswith('site_') or col.startswith('sex_') or col == 'age_normalized']
    for col in metadata_columns:
        if col == 'age_normalized':
            df[col] = df[col].fillna(0.5)  # Default to middle age
        else:
            df[col] = df[col].fillna(0)  # Default to 0 for one-hot encoded columns
    
    # Create a diagnosis column based on the one-hot encoded columns
    diagnosis_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    # Convert to numeric diagnosis index (0-8)
    df['diagnosis'] = df[diagnosis_columns].idxmax(axis=1)
    
    # Create a mapping from diagnosis name to index
    diagnosis_to_idx = {name: idx for idx, name in enumerate(diagnosis_columns)}
    
    # Count samples per class
    class_counts = df['diagnosis'].value_counts()
    print(f"Class distribution before filtering: {class_counts}")
    
    # Filter out classes with too few samples
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    df = df[df['diagnosis'].isin(valid_classes)]
    
    # Update the mapping to only include valid classes
    diagnosis_to_idx = {name: idx for idx, name in enumerate(diagnosis_columns) if name in valid_classes}
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_indices = []
    
    for train_idx, val_idx in skf.split(df, df['diagnosis']):
        fold_indices.append({
            "train_idx": train_idx,
            "val_idx": val_idx
        })
    
    print(f"Created {n_folds} folds with stratification")
    print(f"Final class mapping: {diagnosis_to_idx}")
    print(f"Final dataset size: {len(df)} samples with metadata features")
    
    return df, fold_indices, diagnosis_to_idx

class YOLODetectionGenerator:
    """
    Custom generator for YOLO detected images with multi-class classification and metadata
    """
    def __init__(self, df, diagnosis_to_idx, batch_size=16, is_training=True, 
                 seed=42, memory_efficient=True, augmentation_strength='medium'):
        self.df = df
        self.diagnosis_to_idx = diagnosis_to_idx
        self.batch_size = batch_size
        self.is_training = is_training
        self.memory_efficient = memory_efficient
        self.augmentation_strength = augmentation_strength
        self.seed = seed
        
        # Extract diagnoses and image paths
        self.diagnoses = df['diagnosis'].values
        self.image_paths = df['image_path'].values
        self.label_paths = df['label_path'].values
        
        # Extract metadata features
        self.metadata_columns = [col for col in df.columns if col.startswith('site_') or col.startswith('sex_') or col == 'age_normalized']
        self.metadata = df[self.metadata_columns].values
        self.metadata_dim = len(self.metadata_columns)
        
        # Number of classes is the number of unique diagnoses
        self.num_classes = len(diagnosis_to_idx)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Calculate number of batches
        self.n_samples = len(df)
        self.indices = np.arange(self.n_samples)
        self.steps_per_epoch = int(np.ceil(self.n_samples / self.batch_size))
        
        # Shuffle indices if training
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return self.steps_per_epoch
    
    def reset(self):
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def augment_image(self, image):
        """Apply data augmentation based on strength setting"""
        if not self.is_training:
            return image
        
        # Define augmentation parameters based on strength
        if self.augmentation_strength == 'light':
            rotation_range = 15
            zoom_range = 0.1
            flip_prob = 0.3
        elif self.augmentation_strength == 'medium':
            rotation_range = 30
            zoom_range = 0.15
            flip_prob = 0.5
        else:  # strong
            rotation_range = 45
            zoom_range = 0.2
            flip_prob = 0.7
        
        # Apply random rotation
        if random.random() < 0.7:
            angle = random.uniform(-rotation_range, rotation_range)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        # Apply random zoom
        if random.random() < 0.5:
            zoom = random.uniform(1.0 - zoom_range, 1.0 + zoom_range)
            width, height = image.size
            new_width = int(width * zoom)
            new_height = int(height * zoom)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            if zoom > 1.0:
                # Zoom in: crop and resize
                image = image.crop((left, top, right, bottom)).resize((width, height), Image.BILINEAR)
            else:
                # Zoom out: resize and pad
                resized = image.resize((new_width, new_height), Image.BILINEAR)
                new_img = Image.new('RGB', (width, height), (0, 0, 0))
                new_img.paste(resized, (left, top))
                image = new_img
        
        # Apply horizontal flip
        if random.random() < flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply color jitter
        if random.random() < 0.5:
            enhancer = random.choice([
                lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
                lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
                lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
            ])
            image = enhancer(image)
        
        return image
    
    def load_and_preprocess_image(self, image_path, label_path):
        """Load image and apply YOLO crop based on label"""
        try:
            # Load the image
            image = Image.open(image_path).convert('RGB')
            img_width, img_height = image.size
            
            # Load YOLO label
            with open(label_path, 'r') as f:
                label_content = f.read().strip().split()
            
            # Parse YOLO format: class x_center y_center width height confidence
            if len(label_content) >= 5:
                x_center = float(label_content[1]) * img_width
                y_center = float(label_content[2]) * img_height
                width = float(label_content[3]) * img_width
                height = float(label_content[4]) * img_height
                
                # Calculate bounding box coordinates
                left = max(0, int(x_center - width / 2))
                top = max(0, int(y_center - height / 2))
                right = min(img_width, int(x_center + width / 2))
                bottom = min(img_height, int(y_center + height / 2))
                
                # Crop the image to the bounding box
                image = image.crop((left, top, right, bottom))
            
            # Resize to 224x224
            image = image.resize((224, 224), Image.BILINEAR)
            
            # Apply augmentation if training
            if self.is_training:
                image = self.augment_image(image)
            
            # Convert to array and preprocess
            img_array = img_to_array(image)
            img_array = preprocess_input(img_array)
            
            return img_array
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return a blank image in case of error
            return np.zeros((224, 224, 3))
    
    def __getitem__(self, idx):
        """Get a batch of data with metadata"""
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_size = len(batch_indices)
        batch_images = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
        batch_metadata = np.zeros((batch_size, self.metadata_dim), dtype=np.float32)
        batch_labels = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        
        # Fill the batch
        for i, idx in enumerate(batch_indices):
            # Load and preprocess image
            img_array = self.load_and_preprocess_image(
                self.image_paths[idx], 
                self.label_paths[idx]
            )
            batch_images[i] = img_array
            
            # Add metadata
            batch_metadata[i] = self.metadata[idx]
            
            # Create one-hot encoded label
            diagnosis = self.diagnoses[idx]
            diagnosis_idx = self.diagnosis_to_idx.get(diagnosis, 0)  # Default to 0 if not found
            batch_labels[i, diagnosis_idx] = 1.0
            
            # Free memory if needed
            if self.memory_efficient and i % 10 == 0:
                gc.collect()
        
        # Return images, metadata, and labels
        return [batch_images, batch_metadata], batch_labels


def create_yolo_generators(
    csv_path,
    image_dir,
    labels_dir,
    batch_size=16,
    min_samples_per_class=5,
    fold_idx=0,
    n_folds=5,
    seed=42,
    augmentation_strength='medium',
    metadata_csv_path= "/kaggle/input/2019-isic-csv/ISIC_2019_Training_Metadata.csv"
):
    """
    Create train and validation generators for a specific fold using YOLO detected images
    Uses 4 folds for training (with augmentation) and 1 fold for testing (no modification)
    
    Args:
        csv_path: Path to the ground truth CSV
        image_dir: Directory containing the images
        labels_dir: Directory containing YOLO labels
        batch_size: Batch size for training
        min_samples_per_class: Minimum samples required for a class to be included
        fold_idx: Index of the fold to use for validation
        n_folds: Number of folds for cross-validation
        seed: Random seed for reproducibility
        augmentation_strength: Strength of data augmentation ('light', 'medium', 'strong')
        metadata_csv_path: Path to the metadata CSV file (optional)
        
    Returns:
        Train generator, validation generator, and class indices
    """
    # Load data with or without metadata
    if metadata_csv_path:
        df, fold_indices, diagnosis_to_idx = load_yolo_detections(
            image_dir, labels_dir, csv_path, metadata_csv_path, min_samples_per_class, n_folds
        )
        print("Using metadata features for training")
    else:
        df, fold_indices, diagnosis_to_idx = load_yolo_detections(
            image_dir, labels_dir, csv_path, None, min_samples_per_class, n_folds
        )
        print("Not using metadata features")

    # Get indices for all folds
    all_train_indices = []
    test_indices = None
    
    # Combine 4 folds for training, use 1 fold for testing
    for i, fold_data in enumerate(fold_indices):
        if i == fold_idx:
            # This is the test fold
            test_indices = fold_data["val_idx"]
        else:
            # Add to training folds
            all_train_indices.extend(fold_data["train_idx"])
    
    # Create train and test dataframes
    train_df = df.iloc[all_train_indices]
    test_df = df.iloc[test_indices]
    
    print(f"Training on {len(train_df)} samples from 4 folds")
    print(f"Testing on {len(test_df)} samples from fold {fold_idx}")

    # Create generators - apply augmentation only to training data
    train_generator = YOLODetectionGenerator(
        train_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=True,  # Apply augmentation
        seed=seed,
        memory_efficient=True,
        augmentation_strength=augmentation_strength  # Pass augmentation strength
    )

    test_generator = YOLODetectionGenerator(
        test_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=False,  # No augmentation for test data
        seed=seed + 1,
        memory_efficient=True
    )

    # Return only what's expected in train.py
    return train_generator, test_generator, diagnosis_to_idx


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
        img_name = os.path.basename(img_path).split(".")[0]

        # Find corresponding label file
        label_path = os.path.join(labels_dir, f"{img_name}.txt")

        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            continue

        # Find the original image entry in the CSV
        original_entry = df[df["image_name"] == img_name]

        # Skip if no matching entry in CSV
        if len(original_entry) == 0:
            continue

        # Read the label file to get bounding box
        with open(label_path, "r") as f:
            label_content = f.read().strip().split()

        # Parse label content (class x_center y_center width height confidence)
        if len(label_content) >= 5:  # Ensure we have at least the basic bbox info
            # Add to detected data
            detected_data.append(
                {
                    "image_name": img_name,
                    "diagnosis": original_entry["diagnosis"].values[0],
                    "benign_malignant": original_entry["benign_malignant"].values[0],
                    "target": original_entry["target"].values[0],
                }
            )

    # Create dataframe from detected data
    detected_df = pd.DataFrame(detected_data)

    print("Dataset Statistics:")
    print("-" * 50)
    print(f"Total number of detected images: {len(detected_df)}")
    print("\nDiagnosis distribution:")
    diagnosis_counts = detected_df["diagnosis"].value_counts()
    print(diagnosis_counts)

    print("\nClasses with less than 2 samples:")
    print(diagnosis_counts[diagnosis_counts < 2])

    print("\nBenign/Malignant distribution:")
    print(detected_df["benign_malignant"].value_counts())

    # Calculate class weights for imbalanced data (only for classes with enough samples)
    valid_classes = diagnosis_counts[diagnosis_counts >= 2].index
    valid_df = detected_df[detected_df["diagnosis"].isin(valid_classes)]

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(valid_df["diagnosis"]), y=valid_df["diagnosis"]
    )

    print("class_weights data_preprocessing", class_weights)
    print(
        "dict(zip(np.unique(valid_df['diagnosis']), class_weights))",
        dict(zip(np.unique(valid_df["diagnosis"]), class_weights)),
    )

    return {
        "class_weights": dict(zip(np.unique(valid_df["diagnosis"]), class_weights)),
        "n_classes": len(valid_classes),
        "dataset_size": len(detected_df),  # Total number of images
        "valid_dataset_size": len(valid_df),  # Number of images in valid classes
    }


# Usage example:
if __name__ == "__main__":
    CSV_PATH = "/kaggle/input/2019-isic-csv/ISIC_2019_Training_GroundTruth.csv"
    IMAGE_DIR = "/kaggle/input/2019-isic/exp"

    # Analyze dataset
    dataset_stats = analyze_dataset(CSV_PATH)
    print("\nClass weights for handling imbalance:")
    print(dataset_stats["class_weights"])

    # Create generators
    train_gen, val_gen, diagnosis_to_idx, n_classes, fold_size = create_generators(
        CSV_PATH, IMAGE_DIR, fold_idx=0
    )


# Add this function to your data_preprocessing.py file

def create_optimized_dataset(generator, is_training=True):
    """
    Convert a Keras Sequence generator to an optimized tf.data.Dataset
    
    Args:
        generator: A Keras Sequence generator (e.g., YOLODetectionGenerator)
        is_training: Whether this dataset is for training (enables shuffling)
        
    Returns:
        A tf.data.Dataset optimized for performance
    """
    # Get the total number of samples
    num_samples = len(generator.indices)
    batch_size = generator.batch_size
    
    # Create a dataset from generator indices
    dataset = tf.data.Dataset.from_tensor_slices(generator.indices)
    
    # Shuffle if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(10000, num_samples), 
                                  reshuffle_each_iteration=True,
                                  seed=generator.seed)
    
    # Map function to load images and labels
    def load_sample(idx):
        # Convert to Python int to avoid TF eager execution issues
        idx = int(idx.numpy())
        
        # Get image path and bbox
        image_path = generator.image_paths[idx]
        bbox = generator.bboxes[idx]
        
        # Get diagnosis and convert to one-hot
        diagnosis = generator.diagnoses[idx]
        diagnosis_idx = generator.diagnosis_to_idx[diagnosis]
        label = tf.one_hot(diagnosis_idx, generator.n_classes)
        
        # Load and preprocess image
        img = Image.open(image_path)
        
        # Apply bounding box cropping
        width, height = img.size
        x_center, y_center, box_width, box_height = bbox
        
        # Convert normalized coordinates to pixel values
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Crop the image
        img = img.crop((x1, y1, x2, y2))
        
        # Resize to target size
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        
        # Apply augmentation if training
        if is_training and generator.augmentation:
            img_array = generator.augmentation(tf.expand_dims(img_array, 0))[0]
        
        return img_array, label
    
    # Wrap the function to handle TF eager execution
    def tf_load_sample(idx):
        img, label = tf.py_function(
            load_sample,
            [idx],
            [tf.float32, tf.float32]
        )
        # Set shapes explicitly
        img.set_shape((224, 224, 3))
        label.set_shape((generator.n_classes,))
        return img, label
    
    # Map the loading function
    dataset = dataset.map(
        tf_load_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
