python yolov5/detect.py \
  --weights lesion_runs/yolov5x_skin_lesions4/weights/best.pt \
  --source first_100_images \
  --conf 0.25 \
  --save-txt \
  --save-conf \
  --save-crop


# --save-txt: Saves bounding box coordinates in .txt files.

# --save-conf: Also saves confidence scores in those .txt files.

# --save-crop: Saves cropped images of each detection.

