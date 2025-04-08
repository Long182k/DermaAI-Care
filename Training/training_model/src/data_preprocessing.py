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
import traceback


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


def create_yolo_generators(csv_path, image_dir, labels_dir, batch_size=32, min_samples_per_class=5,
                         n_folds=5, fold_idx=0, seed=42, augmentation_strength='medium',
                         metadata_csv_path=None):
    """
    Create data generators for YOLO format data with optimized tf.data pipeline
    
    Args:
        csv_path: Path to CSV file with image metadata
        image_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        batch_size: Batch size for training
        min_samples_per_class: Minimum samples per class for stratification
        n_folds: Number of folds for cross-validation
        fold_idx: Index of the fold to use for validation
        seed: Random seed for reproducibility
        augmentation_strength: Strength of data augmentation ('low', 'medium', 'high')
        metadata_csv_path: Path to CSV file with patient metadata
    
    Returns:
        train_generator, val_generator, diagnosis_to_idx, n_classes, n_folds, class_weights
    """
    try:
        # Set data format explicitly
        tf.keras.backend.set_image_data_format('channels_last')
        
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Determine ID column - could be 'image_id', 'image', or other
        id_column = None
        for col in ['image_id', 'image', 'isic_id', 'id']:
            if col in df.columns:
                id_column = col
                break
        
        if id_column is None:
            # If no ID column found, create one from the first column
            df['image_id'] = df.iloc[:, 0]
            id_column = 'image_id'
            print(f"No ID column found, using first column '{df.columns[0]}' as image_id")
        
        # Determine diagnosis column - could be 'dx', 'diagnosis', or we need to use multi-label columns
        dx_column = None
        for col in ['dx', 'diagnosis', 'class']:
            if col in df.columns:
                dx_column = col
                break
        
        # Check if we're dealing with multi-label ISIC format
        isic_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        isic_format = all(col in df.columns for col in isic_classes[:3])  # Check at least a few classes exist
        
        if isic_format:
            print("Detected ISIC 2019 multi-label format")
            
            # For multi-label, create a dummy 'dx' column with the highest probability class
            class_columns = [col for col in isic_classes if col in df.columns]
            
            # Create a new column with the class that has the highest probability
            df['dx'] = df[class_columns].idxmax(axis=1)
            dx_column = 'dx'
            
            # Create mapping for diagnosis to index
            diagnosis_to_idx = {label: i for i, label in enumerate(class_columns)}
            n_classes = len(diagnosis_to_idx)
            
            # Use one-hot encoding for targets
            # Create a new column with multi-hot encoding for all diagnoses
            df['target'] = df.apply(lambda row: [row[col] for col in class_columns], axis=1)
            
            print(f"Created multi-label targets from columns: {class_columns}")
        elif dx_column:
            # Single label classification
            diagnosis_to_idx = {d: i for i, d in enumerate(sorted(df[dx_column].unique()))}
            n_classes = len(diagnosis_to_idx)
            
            # One-hot encode the diagnoses
            df['target'] = df[dx_column].apply(lambda x: tf.keras.utils.to_categorical(
                diagnosis_to_idx[x], num_classes=n_classes).tolist())
        else:
            raise ValueError("Could not determine diagnosis column. Please check your CSV format.")
        
        print(f"Found {n_classes} classes: {diagnosis_to_idx}")
        
        # Calculate class weights for imbalanced data
        if dx_column:
            class_counts = df[dx_column].value_counts()
            total_samples = len(df)
            class_weights = {i: total_samples / (n_classes * count) 
                            for i, count in enumerate(class_counts)}
        else:
            # For multi-label, use a default weight of 1.0 for all classes
            class_weights = {i: 1.0 for i in range(n_classes)}
        
        # Create stratified k-fold split for better cross-validation
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        # For stratification, use the class with highest probability in multi-label case
        if isic_format:
            stratify_column = df['dx']
        else:
            stratify_column = df[dx_column]
            
        fold_indices = list(kf.split(df, stratify_column))
        
        # Get train and validation indices for the current fold
        train_idx, val_idx = fold_indices[fold_idx]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
        
        # Define augmentation functions
        def parse_image(img_path, label):
            # Load and preprocess the image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32) / 255.0
            return img, label
        
        def augment(image, label):
            # Data augmentation
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            
            # Apply rotation
            angle = tf.random.uniform([], -0.2, 0.2)
            image = tf.image.rot90(image, k=tf.cast(angle * 2, tf.int32))
            
            return image, label
        
        # Create training dataset
        train_img_paths = []
        for img_id in train_df[id_column]:
            # Try different file extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                path = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(path):
                    train_img_paths.append(path)
                    break
            else:
                # If no file found, use a default path and hope it exists
                train_img_paths.append(os.path.join(image_dir, f"{img_id}.jpg"))
        
        train_labels = train_df['target'].tolist()
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))
        train_dataset = train_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset
        val_img_paths = []
        for img_id in val_df[id_column]:
            # Try different file extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                path = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(path):
                    val_img_paths.append(path)
                    break
            else:
                # If no file found, use a default path and hope it exists
                val_img_paths.append(os.path.join(image_dir, f"{img_id}.jpg"))
        
        val_labels = val_df['target'].tolist()
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths, val_labels))
        val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        # For compatibility with the existing code
        class DatasetWrapper:
            def __init__(self, dataset, n_samples, batch_size, n_classes):
                self.dataset = dataset
                self.n_samples = n_samples
                self.batch_size = batch_size
                self.n_classes = n_classes
                self.class_indices = {v: k for k, v in diagnosis_to_idx.items()}
            
            def __len__(self):
                return self.n_samples // self.batch_size
            
            def __getitem__(self, idx):
                # This is just a placeholder to maintain compatibility
                # The actual data comes from the dataset
                return None, None

        # Check if any image paths exist before proceeding
        sample_train_path = train_img_paths[0] if train_img_paths else None
        sample_val_path = val_img_paths[0] if val_img_paths else None
        
        if not sample_train_path or not os.path.exists(sample_train_path):
            print(f"Warning: Sample train image path does not exist: {sample_train_path}")
            print(f"Image directory: {image_dir}")
            print(f"Sample ID: {train_df[id_column].iloc[0]}")
        else:
            print(f"Verified train image path exists: {sample_train_path}")
            
        if not sample_val_path or not os.path.exists(sample_val_path):
            print(f"Warning: Sample val image path does not exist: {sample_val_path}")
        else:
            print(f"Verified val image path exists: {sample_val_path}")
        
        train_wrapper = DatasetWrapper(train_dataset, len(train_df), batch_size, n_classes)
        val_wrapper = DatasetWrapper(val_dataset, len(val_df), batch_size, n_classes)
        
        return train_dataset, val_dataset, diagnosis_to_idx, n_classes, n_folds, class_weights
        
    except Exception as e:
        print(f"Error creating data generators: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None


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
    CSV_PATH = "/kaggle/input/annotated-isic-2019-images/ISIC_2019_Training_GroundTruth.csv"
    IMAGE_DIR = "/kaggle/input/annotated-isic-2019-images/exp/exp"

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
