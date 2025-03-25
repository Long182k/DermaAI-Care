import os

# 1. Path to your images folder
images_dir = "first_100_images"

# 2. Path to YOLOv5 detection labels folder
labels_dir = "yolov5/runs/detect/exp4/labels"

# 3. Gather image filenames (without extension)
valid_exts = {".jpg", ".jpeg", ".png"}
all_images = []
for f in os.listdir(images_dir):
    ext = os.path.splitext(f)[1].lower()
    if ext in valid_exts:
        # strip extension, e.g. "ISIC_0153549.jpg" -> "ISIC_0153549"
        all_images.append(os.path.splitext(f)[0])

# 4. Gather label filenames (without extension)
all_labels = []
if os.path.isdir(labels_dir):
    for f in os.listdir(labels_dir):
        if f.endswith(".txt"):
            all_labels.append(os.path.splitext(f)[0])
else:
    print(f"Labels directory not found: {labels_dir}")
    exit()

# 5. Find which images had no labels
no_detection = []
for img_name in all_images:
    if img_name not in all_labels:
        no_detection.append(img_name)

# 6. Print results
if no_detection:
    print("Images with NO detections (no label file):")
    for nd in no_detection:
        print(nd)
else:
    print("All images had at least one detection.")
