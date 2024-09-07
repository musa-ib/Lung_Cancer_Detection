from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

img_path = 'C:/Users/Lenovo/Desktop/The IQ-OTHNCCD lung cancer dataset/Bengin/Bengin case (11).jpg'
img_path1 = 'C:/Users/Lenovo/Desktop/The IQ-OTHNCCD lung cancer dataset/Malignant/Malignant case (110).jpg'
img_path2 = 'C:/Users/Lenovo/Desktop/The IQ-OTHNCCD lung cancer dataset/Normal/Normal case (21).jpg'
img_path3 = 'C:/Users/Lenovo/Desktop/The IQ-OTHNCCD lung cancer dataset/Bengin/Bengin case (33).jpg'

# model = load_model('lung_cancer_detector.keras')
model = load_model('lung_cancer_VGG19.keras')
class_labels = ['benign', 'malignant', 'normal']

def load_img(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]

# Load your model (if not already loaded)


# Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)
    return class_labels[predicted_class[0]]



# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize to [0, 1]


# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# List of class labels
  # Ensure this matches your model's class order
predicted_label = class_labels[predicted_class[0]]
print(predicted_class)
print(predicted_label)

print(f"Predicted Class: {predicted_label}")
print(load_img(img_path))
print(load_img(img_path1))
print(load_img(img_path2))
print(load_img(img_path3))
