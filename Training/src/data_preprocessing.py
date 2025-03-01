import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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

def create_augmentation_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

def validate_image(image_path):
    """
    Validate if image meets the requirements
    """
    try:
        img = load_img(image_path)
        
        # Check image dimensions
        if img.size != (299, 299):
            print(f"Warning: Image size {img.size} will be resized to (299, 299)")
        
        # Check color channels
        img_array = img_to_array(img)
        if img_array.shape[-1] != 3:
            raise ValueError("Image must be in RGB format (3 channels)")
            
        return True
        
    except Exception as e:
        print(f"Error validating image: {str(e)}")
        return False

def check_dataset_quality(data_dir):
    """
    Check quality of the entire dataset
    """
    issues = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        if img.mode != 'RGB':
                            issues.append(f"{image_path}: Not in RGB mode")
                        if img.size[0] < 299 or img.size[1] < 299:
                            issues.append(f"{image_path}: Smaller than required 299x299")
                        img.verify()
                except Exception as e:
                    issues.append(f"{image_path}: Corrupted or invalid image - {str(e)}")
    return issues

def load_and_prepare_data(csv_path, image_dir, min_samples_per_class=2):
    """
    Load and prepare data from ISIC 2020 dataset
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create full image paths
    df['image_path'] = df['image_name'].apply(lambda x: os.path.join(image_dir, x + '.jpg'))
    
    # Verify all images exist
    missing_images = df[~df['image_path'].apply(os.path.exists)]['image_name'].tolist()
    if missing_images:
        print(f"Warning: {len(missing_images)} images are missing")
    
    # Remove rows with missing images
    df = df[df['image_path'].apply(os.path.exists)]
    
    # Count samples per class
    class_counts = df['diagnosis'].value_counts()
    print("\nSamples per class before filtering:")
    print(class_counts)
    
    # Filter out classes with too few samples
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df['diagnosis'].isin(valid_classes)]
    
    print("\nSamples per class after filtering:")
    print(df['diagnosis'].value_counts())
    
    # Create class mappings for diagnosis
    diagnosis_to_idx = {diagnosis: idx for idx, diagnosis in enumerate(df['diagnosis'].unique())}
    
    print(f"\nNumber of classes: {len(diagnosis_to_idx)}")
    print("Classes:", list(diagnosis_to_idx.keys()))
    
    try:
        # Split data into train and validation
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2,
            stratify=df['diagnosis'],
            random_state=42
        )
        
        print(f"\nTraining set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        
        return train_df, val_df, diagnosis_to_idx
        
    except Exception as e:
        print(f"\nError in train-test split: {str(e)}")
        print("Dataset statistics:")
        print(df['diagnosis'].value_counts())
        raise

class ISICDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, diagnosis_to_idx, batch_size=16, is_training=True):
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
            # Load and resize image to 224x224
            img = load_img(self.image_paths[idx], target_size=(224, 224))
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

def create_generators(csv_path, image_dir, batch_size=16, min_samples_per_class=2):
    """
    Create train and validation generators with optimized memory usage
    """
    train_df, val_df, diagnosis_to_idx = load_and_prepare_data(
        csv_path, 
        image_dir,
        min_samples_per_class=min_samples_per_class
    )
    
    # Create generators with smaller batch size
    train_generator = ISICDataGenerator(
        train_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=True
    )
    
    val_generator = ISICDataGenerator(
        val_df,
        diagnosis_to_idx,
        batch_size=batch_size,
        is_training=False
    )
    
    n_classes = len(diagnosis_to_idx)
    return train_generator, val_generator, diagnosis_to_idx, n_classes

def analyze_dataset(csv_path):
    """
    Analyze dataset statistics
    """
    df = pd.read_csv(csv_path)
    
    print("Dataset Statistics:")
    print("-" * 50)
    print(f"Total number of images: {len(df)}")
    print("\nDiagnosis distribution:")
    diagnosis_counts = df['diagnosis'].value_counts()
    print(diagnosis_counts)
    
    print("\nClasses with less than 2 samples:")
    print(diagnosis_counts[diagnosis_counts < 2])
    
    print("\nBenign/Malignant distribution:")
    print(df['benign_malignant'].value_counts())
    print("\nAnatomical site distribution:")
    print(df['anatom_site_general_challenge'].value_counts())
    
    # Calculate class weights for imbalanced data (only for classes with enough samples)
    valid_classes = diagnosis_counts[diagnosis_counts >= 2].index
    valid_df = df[df['diagnosis'].isin(valid_classes)]
    
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
    CSV_PATH = "data/ISIC_2020_Training_GroundTruth_v2.csv"
    IMAGE_DIR = "data/train"
    
    # Analyze dataset
    dataset_stats = analyze_dataset(CSV_PATH)
    print("\nClass weights for handling imbalance:")
    print(dataset_stats['class_weights'])
    
    # Create generators
    train_gen, val_gen, diagnosis_to_idx, n_classes = create_generators(
        CSV_PATH,
        IMAGE_DIR,
        batch_size=16
    ) 