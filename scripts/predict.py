import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('models/fpt_plus_model.keras')

# Define a function to preprocess the image and make a prediction
def predict_image(test_image_path):
    img = image.load_img(test_image_path, target_size=(150, 150))  # Use the same size as during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if required (match training preprocessing)

    predictions = model.predict(img_array)
    class_labels = ['class1', 'class2']  # Update with your class labels
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f'Predicted class: {predicted_class}, Confidence: {confidence:.2f}')

# Path to your test image
test_image_path = 'path_to_your_test_image.jpg'
predict_image(test_image_path)
