# # 1. Clone the official YOLOv5 repository (if not already)
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# # 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm your dataset is in the right place
#    For example, after your Roboflow snippet, you should have a folder "SkinLesionDetection-4"
#    with subfolders: "train/images", "train/labels", "valid/images", "valid/labels"
#    and a data.yaml file. Let's assume data.yaml is at: 'SkinLesionDetection-4/data.yaml'

# 4. Train YOLOv5x
# python yolov5/train.py \
#   --data /teamspace/studios/this_studio/Skin-Lesion-Detection-4/data.yaml \
#   --weights yolov5x.pt \
#   --img 640 \
#   --batch 16 \
#   --epochs 100 \
#   --project "lesion_runs" \
#   --name "yolov5x_skin_lesions"

# Explanation of the arguments:
# --data: Path to the data.yaml that points to your train/val sets
# --weights: Starting checkpoint (yolov5x.pt is the YOLOv5x pretrained model)
# --img: Training image size (640 is standard)
# --batch: Adjust based on GPU memory
# --epochs: Number of training epochs
# --project and --name: Where to save training results

# 5. After training completes, the best model weights will be in:
#    lesion_runs/yolov5x_skin_lesions/weights/best.pt

# 6. (Optional) Run inference on test images
#    You can do something like:
# !python detect.py \
#   --weights lesion_runs/yolov5x_skin_lesions/weights/best.pt \
#   --source ../SkinLesionDetection-4/test/images \
#   --conf 0.25 \
#   --name "test_predictions" \
#   --project "lesion_runs"

# 7. The predictions with bounding boxes will appear in:
#    lesion_runs/test_predictions
