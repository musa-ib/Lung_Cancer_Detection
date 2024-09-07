import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your model
model = load_model('lung_cancer_VGG19.keras')

# List of class labels
class_labels = ['benign', 'malignant', 'normal']

def predict_image(img):
    # Preprocess the image
    img = img.resize((224, 224))  # Resize image to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    # print(predicted_class)
    return class_labels[predicted_class]

# Streamlit UI
st.title('Lung Cancer Detection')
st.write('Upload an image of a lung scan to predict if it is benign, malignant, or normal.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Predict and display result
    img = Image.open(uploaded_file)
    result = predict_image(img)
    st.write(f"Predicted Class: {result}")
