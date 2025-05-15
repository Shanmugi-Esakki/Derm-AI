import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_dir="raw_data", output_dir="data", split_ratio=(0.7, 0.15, 0.15)):
    # Check if raw_data exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Directory '{source_dir}' not found. Create it and add class subfolders with images.")

    # List class folders
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    if not classes:
        raise ValueError(f"No class subfolders found in '{source_dir}'. Expected folders like 'Melanoma/', 'Nevus/', etc.")

    print(f"Found classes: {classes}")

    # Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            print(f"⚠️ Warning: No images found in {class_path}. Skipping.")
            continue

        print(f"Processing {class_name} ({len(images)} images)...")

        # Split into train/val/test (70/15/15)
        train, temp = train_test_split(images, train_size=split_ratio[0], random_state=42)
        val, test = train_test_split(temp, test_size=split_ratio[2]/(split_ratio[1] + split_ratio[2]), random_state=42)

        # Copy files
        for split, imgs in [("train", train), ("val", val), ("test", test)]:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, split, class_name, img))

    print("✅ Data splitting complete!")

if __name__ == "__main__":
    split_data()