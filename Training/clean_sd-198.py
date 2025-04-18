import os
import zipfile
import shutil

input_dir = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/SD-198"
cleaned_dir = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/SD-198_cleaned"
zip_path = "/Users/drake/Documents/UWE/IT PROJECT/Code/DermaAI-Care/Training/SD_198_cleaned.zip"

# Clean up any previous run
shutil.rmtree(cleaned_dir, ignore_errors=True)

# Copy and rename
for root, dirs, files in os.walk(input_dir):
    for name in files:
        src_path = os.path.join(root, name)
        rel_path = os.path.relpath(src_path, input_dir)

        # Clean the path
        clean_rel_path = rel_path.replace("'", "").replace('"', "")
        dst_path = os.path.join(cleaned_dir, clean_rel_path)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Copy image
        shutil.copy2(src_path, dst_path)

# Zip the cleaned folder
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(cleaned_dir):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, cleaned_dir)
            zipf.write(filepath, arcname)

print("Cleaned and zipped dataset saved at:", zip_path)
