import os
import shutil
import random

# Define paths
data_dir = 'C:\\FTPPLUS\\images'  # Replace with your actual path
train_dir = 'data/train'
val_dir = 'data/validation'
classes = ['class1', 'class2']  # Replace with your actual class names

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Get all images for this class
    class_path = os.path.join(data_dir, class_name)
    images = os.listdir(class_path)

    # Shuffle and split images into train and validation sets (80-20 split)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Move images to train directory
    for img in train_images:
        shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    # Move images to validation directory
    for img in val_images:
        shutil.move(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print("Images have been organized into train and validation directories.")
