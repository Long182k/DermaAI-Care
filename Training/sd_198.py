from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm

# Load the dataset (in memory, from Hugging Face cache)
dataset = load_dataset("resyhgerwshshgdfghsdfgh/SD-198", split="train")

# Optional: get label names
label_names = dataset.features["label"].names

# Define output dir
output_dir = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/SD-198"
os.makedirs(output_dir, exist_ok=True)

# Save each image into folders
for i, example in tqdm(enumerate(dataset), total=len(dataset)):
    label_id = example["label"]
    label_name = label_names[label_id]
    image = example["image"]

    # Create folder for label
    label_folder = os.path.join(output_dir, label_name)
    os.makedirs(label_folder, exist_ok=True)

    # Save image
    image.save(os.path.join(label_folder, f"{i}.jpg"))
