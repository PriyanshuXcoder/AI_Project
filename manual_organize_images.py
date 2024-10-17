import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
data_dir = 'C:\\FTPplus\\images'  # Path to the folder containing all images
train_dir = 'C:\\FTPplus\\data\\train'
val_dir = 'C:\\FTPplus\\data\\validation'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through all class folders inside data_dir
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        # Create class subdirectories in train and validation folders
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # List all images in the class directory
        images = os.listdir(class_path)
        print(f"Found {len(images)} images for class '{class_name}' in {class_path}")
        images = [os.path.join(class_path, img) for img in images]

        # Split into training and validation sets (80% train, 20% validation)
        if len(images) > 5:
            train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        else:
            train_images = images
            val_images = []

        # Move images to respective directories
        for img in train_images:
            shutil.copy(img, train_class_dir)
        for img in val_images:
            shutil.copy(img, val_class_dir)

        print(f"Processed {class_name}: {len(train_images)} training, {len(val_images)} validation")
