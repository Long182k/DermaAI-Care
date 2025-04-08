import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
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
        
        # Pre-compute image paths and targets to avoid memory leaks
        self.image_paths = self.df["image_path"].values
        self.targets = np.array(self.df["target"].tolist())  # Convert list of lists to numpy array
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
            rng = np.random.RandomState(self.seed + idx)
            extra_needed = self.batch_size - len(batch_indices)
            extra_indices = rng.choice(batch_indices, size=extra_needed, replace=True)
            batch_indices = np.concatenate([batch_indices, extra_indices])

        # Initialize batch arrays
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

                # Apply preprocessing
                img_array = preprocess_input(img_array)

                batch_x[i] = img_array
                batch_y[i] = self.targets[idx]  # Use pre-computed targets

                # Memory optimization: explicitly delete arrays
                if self.memory_efficient:
                    del img_array
                    if i % 8 == 0:
                        gc.collect()
            except Exception as e:
                print(f"Error processing image {self.image_paths[idx]}: {e}")
                # Use the previous successful sample in case of error
                if i > 0:
                    batch_x[i] = batch_x[i-1]
                    batch_y[i] = batch_y[i-1]

        return batch_x, batch_y


def load_yolo_detections(
    image_dir, labels_dir, csv_path, min_samples_per_class=5, n_folds=5, metadata_csv_path=None
):
    """
    Load image data and YOLO format detections, combining with metadata from CSV
    
    Args:
        image_dir: Directory containing image files
        labels_dir: Directory containing YOLO label files
        csv_path: Path to CSV file with ground truth labels
        min_samples_per_class: Minimum samples required per class
        n_folds: Number of cross-validation folds
        metadata_csv_path: Path to CSV file with metadata (optional)
    """
    # Load CSV files
    df_labels = pd.read_csv(csv_path)
    
    # Load metadata if provided
    if metadata_csv_path and os.path.exists(metadata_csv_path):
        df_metadata = pd.read_csv(metadata_csv_path)
        # Merge with labels
        df_labels = df_labels.merge(df_metadata, on='image', how='left')
        print(f"Loaded metadata from {metadata_csv_path}")
        print(f"Sample metadata columns: {list(df_metadata.columns)}")
    else:
        print("No metadata file provided or file not found.")
    
    # Define the target classes (ISIC 2019 format)
    target_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    # Create diagnosis mapping
    diagnosis_to_idx = {label: idx for idx, label in enumerate(target_columns)}
    
    # Process each image and its corresponding YOLO label
    image_data = []
    for idx, row in df_labels.iterrows():
        image_name = row['image']
        image_path = os.path.join(image_dir, f"{image_name}.jpg")
        label_path = os.path.join(labels_dir, f"{image_name}.txt")
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            continue
            
        # Get YOLO format bounding box
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    bbox_line = f.readline().strip()
                    if bbox_line:
                        # Split the line and take only the first 5 values
                        values = bbox_line.split()
                        if len(values) >= 5:
                            # Extract class_id and bbox coordinates
                            class_id = float(values[0])
                            x_center = float(values[1])
                            y_center = float(values[2])
                            width = float(values[3])
                            height = float(values[4])
                            bbox = [x_center, y_center, width, height]
                        else:
                            print(f"Warning: Invalid format in {label_path}, using default bbox")
                            bbox = [0.5, 0.5, 1.0, 1.0]  # Default to full image
                    else:
                        bbox = [0.5, 0.5, 1.0, 1.0]  # Default to full image
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
                bbox = [0.5, 0.5, 1.0, 1.0]  # Default to full image
        else:
            bbox = [0.5, 0.5, 1.0, 1.0]  # Default to full image
            
        try:
            # Get the multi-label target vector
            target = [float(row[col]) for col in target_columns]
            
            # Extract metadata if available
            metadata = {}
            if 'age_approx' in row:
                metadata['age_approx'] = row['age_approx'] if not pd.isna(row['age_approx']) else -1.0
            if 'anatom_site_general' in row:
                metadata['anatom_site_general'] = row['anatom_site_general'] if not pd.isna(row['anatom_site_general']) else "unknown"
            if 'sex' in row:
                metadata['sex'] = row['sex'] if not pd.isna(row['sex']) else "unknown"
            
            image_data.append({
                'image_path': image_path,
                'bbox': bbox,
                'target': target,
                'image_name': image_name,
                'metadata': metadata
            })
        except Exception as e:
            print(f"Error processing row for image {image_name}: {e}")
            continue
    
    if not image_data:
        raise ValueError("No valid images found after processing. Please check your data paths and file formats.")
    
    # Convert to DataFrame
    df_processed = pd.DataFrame(image_data)
    
    # Filter out classes with too few samples if needed
    if min_samples_per_class > 0:
        # Convert targets to numpy array for analysis
        targets = np.array([x['target'] for x in image_data])
        # Count samples per class
        class_counts = np.sum(targets, axis=0)
        # Only keep classes with enough samples
        valid_classes = [i for i, count in enumerate(class_counts) if count >= min_samples_per_class]
        
        if not valid_classes:
            raise ValueError(f"No classes have at least {min_samples_per_class} samples. Please reduce min_samples_per_class.")
        
        # Update diagnosis_to_idx to only include valid classes
        diagnosis_to_idx = {label: i for i, (label, _) in enumerate(
            [(label, idx) for label, idx in diagnosis_to_idx.items() 
             if idx in valid_classes]
        )}
        
        # Update targets to only include valid classes
        df_processed['target'] = df_processed['target'].apply(
            lambda x: [x[i] for i in valid_classes]
        )
    
    print(f"\nLoaded {len(df_processed)} images")
    print(f"Number of classes: {len(diagnosis_to_idx)}")
    print("\nClass mapping:")
    for label, idx in diagnosis_to_idx.items():
        print(f"{label}: {idx}")
    
    # Print sample of the metadata
    if df_processed.iloc[0].get('metadata', None):
        print("\nSample metadata:")
        print(df_processed.iloc[0]['metadata'])
    
    # Print a sample of the data to verify
    print("\nSample of loaded data:")
    sample_idx = np.random.randint(0, len(df_processed))
    sample = df_processed.iloc[sample_idx]
    print(f"Image path: {sample['image_path']}")
    print(f"Bounding box: {sample['bbox']}")
    print(f"Target: {sample['target']}")
    
    return df_processed, diagnosis_to_idx


def create_yolo_generators(
    csv_path,
    image_dir,
    labels_dir,
    batch_size=16,
    min_samples_per_class=5,
    n_folds=5,
    fold_idx=0,
    seed=42,
    augmentation_strength='medium',
    metadata_csv_path=None
):
    """
    Create data generators for training and validation using YOLO format detections
    
    Args:
        csv_path: Path to CSV file with ground truth labels
        image_dir: Directory containing image files
        labels_dir: Directory containing YOLO label files
        batch_size: Batch size for training
        min_samples_per_class: Minimum samples required per class
        n_folds: Number of cross-validation folds
        fold_idx: Index of fold to use for validation
        seed: Random seed for reproducibility
        augmentation_strength: Strength of data augmentation ('low', 'medium', 'high')
        metadata_csv_path: Path to CSV file with metadata (optional)
    """
    # Load and preprocess the data
    df_processed, diagnosis_to_idx = load_yolo_detections(
        image_dir, labels_dir, csv_path, min_samples_per_class, n_folds, metadata_csv_path
    )
    
    # Create stratified k-fold split based on the dominant class for each sample
    dominant_classes = df_processed['target'].apply(lambda x: np.argmax(x))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(df_processed, dominant_classes))
    
    # Get train and validation indices for the current fold
    train_idx, val_idx = fold_indices[fold_idx]
    
    # Split the data into training and validation sets
    train_df = df_processed.iloc[train_idx].reset_index(drop=True)
    val_df = df_processed.iloc[val_idx].reset_index(drop=True)
    
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
    
    # Calculate class weights for training
    train_targets = np.array(train_df['target'].tolist())
    class_weights = {}
    for i in range(len(diagnosis_to_idx)):
        # Calculate weight for each class based on positive vs negative samples
        n_pos = np.sum(train_targets[:, i])
        n_neg = len(train_df) - n_pos
        
        # Avoid division by zero
        if n_pos > 0:
            pos_weight = n_neg / n_pos
        else:
            pos_weight = 1.0
            
        # Cap weights to prevent extreme values
        pos_weight = min(max(pos_weight, 0.1), 10.0)
        
        class_weights[i] = pos_weight
    
    print(f"Class weights: {class_weights}")
    
    # Create data generators
    train_generator = YOLODetectionGenerator(
        train_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=True,
        seed=seed,
        augmentation_strength=augmentation_strength
    )
    
    val_generator = YOLODetectionGenerator(
        val_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=False,
        seed=seed
    )
    
    return train_generator, val_generator, diagnosis_to_idx, len(diagnosis_to_idx), n_folds, class_weights


def analyze_yolo_dataset(csv_path, image_dir, labels_dir, metadata_csv_path=None):
    """
    Analyze the dataset and compute class weights and other statistics for multi-label classification
    
    Args:
        csv_path: Path to CSV file with ground truth labels
        image_dir: Directory containing image files
        labels_dir: Directory containing YOLO label files
        metadata_csv_path: Path to CSV file with metadata (optional)
    """
    # Load the data
    df_processed, diagnosis_to_idx = load_yolo_detections(
        image_dir, labels_dir, csv_path, min_samples_per_class=0, metadata_csv_path=metadata_csv_path
    )
    
    # Convert targets to numpy array for analysis
    targets = np.array([x for x in df_processed['target'].values])
    
    # Calculate class distribution
    class_distribution = np.sum(targets, axis=0)
    total_samples = len(df_processed)
    
    # Calculate class weights using balanced approach
    class_weights = {}
    for i in range(len(diagnosis_to_idx)):
        n_pos = np.sum(targets[:, i])
        n_neg = total_samples - n_pos
        # Handle case where a class might have no positive samples
        if n_pos > 0:
            pos_weight = n_neg / n_pos
        else:
            pos_weight = 1.0
        class_weights[i] = pos_weight
    
    # Calculate co-occurrence matrix
    co_occurrence = np.zeros((len(diagnosis_to_idx), len(diagnosis_to_idx)))
    for target in targets:
        for i in range(len(diagnosis_to_idx)):
            for j in range(len(diagnosis_to_idx)):
                if target[i] == 1 and target[j] == 1:
                    co_occurrence[i, j] += 1
    
    # Calculate statistics
    stats = {
        'total_samples': total_samples,
        'class_distribution': {
            label: int(class_distribution[i])
            for label, i in diagnosis_to_idx.items()
        },
        'class_weights': class_weights,
        'co_occurrence_matrix': co_occurrence.tolist(),
        'class_mapping': diagnosis_to_idx,
        'multi_label_stats': {
            'avg_labels_per_sample': float(np.mean(np.sum(targets, axis=1))),
            'max_labels_per_sample': int(np.max(np.sum(targets, axis=1))),
            'samples_with_multiple_labels': int(np.sum(np.sum(targets, axis=1) > 1))
        }
    }
    
    # Check for metadata
    has_metadata = 'metadata' in df_processed.columns and len(df_processed) > 0 and df_processed.iloc[0].get('metadata')
    if has_metadata:
        stats['has_metadata'] = True
        print("\nMetadata is available and will be used for analysis")
    else:
        stats['has_metadata'] = False
        print("\nNo metadata available or provided")
    
    # Print analysis
    print("\nDataset Analysis:")
    print(f"Total number of samples: {total_samples}")
    print("\nClass distribution:")
    for label, count in stats['class_distribution'].items():
        percentage = (count / total_samples) * 100
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    
    print("\nMulti-label statistics:")
    print(f"Average labels per sample: {stats['multi_label_stats']['avg_labels_per_sample']:.2f}")
    print(f"Maximum labels per sample: {stats['multi_label_stats']['max_labels_per_sample']}")
    print(f"Samples with multiple labels: {stats['multi_label_stats']['samples_with_multiple_labels']} ({(stats['multi_label_stats']['samples_with_multiple_labels']/total_samples)*100:.2f}%)")
    
    print("\nClass weights:")
    for label, idx in diagnosis_to_idx.items():
        print(f"{label}: {class_weights[idx]:.2f}")
    
    return stats


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
