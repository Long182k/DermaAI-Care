import os

# Define the path to the annotated images folder
folder_path = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/data/isic-2019/labels"

# List all files in the folder and count those that are files
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(path=os.path.join(folder_path, f))]
image_count = len(image_files)

print(f"Number of images in '{folder_path}': {image_count}")
