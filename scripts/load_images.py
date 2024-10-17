import os
from PIL import Image
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(img)
    return images

def images_to_numpy(images):
    return np.array([np.array(image) for image in images])

if __name__ == "__main__":
    # Load training images
    train_images_class1 = load_images_from_folder('data/train/class1')
    train_images_class2 = load_images_from_folder('data/train/class2')

    # Load validation images
    val_images_class1 = load_images_from_folder('data/validation/class1')
    val_images_class2 = load_images_from_folder('data/validation/class2')

    # Convert images to numpy arrays
    train_images_class1_np = images_to_numpy(train_images_class1)
    train_images_class2_np = images_to_numpy(train_images_class2)
    val_images_class1_np = images_to_numpy(val_images_class1)
    val_images_class2_np = images_to_numpy(val_images_class2)

    print(f'Loaded {len(train_images_class1)} training images from class1.')
    print(f'Loaded {len(train_images_class2)} training images from class2.')
    print(f'Loaded {len(val_images_class1)} validation images from class1.')
    print(f'Loaded {len(val_images_class2)} validation images from class2.')
