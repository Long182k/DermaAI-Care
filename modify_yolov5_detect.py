import os
import shutil
import re

# 1. First, create a modified version of the detect.py script
# We'll create a copy of the original script with our modifications

def modify_yolov5_detect():
    # Read the original detect.py
    with open('/kaggle/working/yolov5/detect.py', 'r') as f:
        content = f.read()
    
    # Create a backup
    with open('/kaggle/working/yolov5/detect_original.py', 'w') as f:
        f.write(content)
    
    # Instead of using regex for complex modifications, let's modify the file line by line
    lines = content.split('\n')
    modified_lines = []
    
    # Flag to track when we're in the save_one_box function
    in_save_one_box = False
    
    for line in lines:
        # Check if we're entering the save_one_box function
        if "def save_one_box(" in line:
            in_save_one_box = True
            modified_lines.append(line)
        # Add directory creation before saving in save_one_box
        elif in_save_one_box and "if save:" in line:
            modified_lines.append(line)
            # Add the directory creation with proper indentation
            indent = line.split("if")[0]  # Get the indentation
            modified_lines.append(f"{indent}    # Create directories if they don't exist")
            modified_lines.append(f"{indent}    os.makedirs(os.path.dirname(file), exist_ok=True)")
        # Modify path handling to preserve full paths
        elif "save_path = str(Path(out) / Path(p).name)" in line:
            indent = line.split("save_path")[0]
            modified_lines.append(f"{indent}save_path = str(Path(out) / p)")
        # Modify the path processing section
        elif "p = Path(p)" in line:
            indent = line.split("p =")[0]
            modified_lines.append(line)
            modified_lines.append(f"{indent}# Use full path structure")
        elif "save_path = str(save_dir / p.name)" in line:
            indent = line.split("save_path")[0]
            modified_lines.append(f"{indent}save_path = str(save_dir / p)")
        # Handle the txt_path line that caused the indentation error
        elif "txt_path = str(save_dir / \"labels\" / p.stem)" in line:
            indent = line.split("txt_path")[0]
            modified_lines.append(f"{indent}txt_path = str(save_dir / \"labels\" / p.stem) + (\"\" if dataset.mode == \"image\" else f\"_{{frame}}\")")
        else:
            modified_lines.append(line)
    
    # Save the modified script
    with open('/kaggle/working/yolov5/detect_modified.py', 'w') as f:
        f.write('\n'.join(modified_lines))
    
    return '/kaggle/working/yolov5/detect_modified.py'

# 2. A function to prepare the paths file for YOLOv5
def prepare_paths_file(data_df):
    # Get the image paths
    img_list = data_df['name'].tolist()
    
    # Write the paths to a file
    with open('/kaggle/working/img_paths.txt', 'w') as f:
        f.write('\n'.join(img_list))
    
    print(f"Wrote {len(img_list)} image paths to /kaggle/working/img_paths.txt")
    return img_list

# 3. Clean any existing detection results
def clean_detection_folder():
    if os.path.exists('/kaggle/working/runs/detect/combined_run'):
        shutil.rmtree('/kaggle/working/runs/detect/combined_run')
    os.makedirs('/kaggle/working/runs/detect/combined_run', exist_ok=True)

# 4. Run the modified detection script
def run_modified_detection(detect_script):
    cmd = f"""python {detect_script} \\
      --weights /kaggle/input/yolo_skin_lesion_detection/pytorch/default/1/best.pt \\
      --source /kaggle/working/img_paths.txt \\
      --conf 0.25 \\
      --save-txt \\
      --save-conf \\
      --save-crop \\
      --project /kaggle/working/runs/detect \\
      --name combined_run \\
      --exist-ok"""
    
    os.system(cmd)

# 5. A fallback solution if the script modification doesn't work as expected
def fallback_solution(data_df):
    # Create a directory structure that mirrors the original
    base_dir = '/kaggle/working/runs/detect/combined_run'
    
    # Make sure crops and labels directories exist
    crops_dir = os.path.join(base_dir, 'crops/skin-lesions')
    labels_dir = os.path.join(base_dir, 'labels')
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create a mapping from simple names to full paths
    path_map = {}
    for full_path in data_df['name']:
        simple_name = os.path.basename(full_path)
        path_map[simple_name] = full_path
        # Also map without extension
        base_name = os.path.splitext(simple_name)[0]
        path_map[base_name] = full_path
    
    # Reorganize the crops
    full_path_crops_dir = os.path.join(base_dir, 'full_path_crops')
    os.makedirs(full_path_crops_dir, exist_ok=True)
    
    # Copy and rename the crop files
    for file in os.listdir(crops_dir):
        if not file.endswith('.jpg'):
            continue
            
        src_path = os.path.join(crops_dir, file)
        if os.path.exists(src_path):
            # Get the base name
            base_name = os.path.splitext(file)[0]
            
            # Find the corresponding full path
            if base_name in path_map:
                full_path = path_map[base_name]
                # Create directory structure
                rel_path = full_path.replace('/kaggle/input/', '')
                full_path_dir = os.path.join(full_path_crops_dir, os.path.dirname(rel_path))
                os.makedirs(full_path_dir, exist_ok=True)
                
                # Copy the file with full path structure
                dst_path = os.path.join(full_path_crops_dir, rel_path)
                shutil.copy2(src_path, dst_path)
                print(f"Copied crop: {src_path} -> {dst_path}")
    
    # Do the same for label files
    full_path_labels_dir = os.path.join(base_dir, 'full_path_labels')
    os.makedirs(full_path_labels_dir, exist_ok=True)
    
    for file in os.listdir(labels_dir):
        if not file.endswith('.txt'):
            continue
            
        src_path = os.path.join(labels_dir, file)
        if os.path.exists(src_path):
            base_name = os.path.splitext(file)[0]
            
            if base_name in path_map:
                full_path = path_map[base_name]
                rel_path = full_path.replace('/kaggle/input/', '')
                full_path_dir = os.path.join(full_path_labels_dir, os.path.dirname(rel_path))
                os.makedirs(full_path_dir, exist_ok=True)
                
                dst_path = os.path.join(full_path_labels_dir, rel_path.replace('.jpg', '.txt'))
                shutil.copy2(src_path, dst_path)
                print(f"Copied label: {src_path} -> {dst_path}")

# Main execution
def process_detection_with_full_paths(data_df):
    # Prepare the paths file
    img_list = prepare_paths_file(data_df)
    
    # Clean any existing detection results
    clean_detection_folder()
    
    try:
        # Try the script modification approach
        detect_script = modify_yolov5_detect()
        run_modified_detection(detect_script)
        print("Detection completed with modified script!")
    except Exception as e:
        print(f"Script modification approach failed: {e}")
        print("Falling back to the standard detection with post-processing...")
        
        # Run the original detection script
        run_modified_detection('/kaggle/working/yolov5/detect.py')
        
        # Apply the fallback solution
        fallback_solution(data_df)
        print("Fallback solution applied. Check the 'full_path_crops' and 'full_path_labels' directories.")
    
    print("Processing complete!")

# Call the main function
process_detection_with_full_paths(data_df)