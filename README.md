# Lung_Cancer_Detection

# Overview

This application allows users to upload images of lung scans and get predictions on whether the image represents a benign, malignant, or normal condition. The application leverages a pre-trained VGG19 model for classification.
Requirements

# To run this application, you'll need the following Python libraries:

    streamlit
    tensorflow
    numpy
    PIL (Pillow)

You can install these libraries using pip:

bash

pip install streamlit tensorflow numpy pillow

# Model

The application uses a pre-trained VGG19 model saved as lung_cancer_VGG19.keras. This model was trained to classify lung scans into three categories:

    Benign
    Malignant
    Normal

# Usage

    Start the Streamlit app:

    Open your terminal and navigate to the directory containing app.py (the name of the script you have) and run:

    bash

    streamlit run app.py

    This will start a local web server and open the app in your default web browser.

    Upload an Image:

    On the web interface, click the "Choose an image..." button to upload a lung scan image in JPG, JPEG, or PNG format.

    View the Prediction:

    After uploading the image, the application will display the uploaded image along with the prediction of whether it is benign, malignant, or normal.

Example

Here is a sample usage flow:

    Upload an image of a lung scan.
    The application displays the uploaded image.
    The model predicts and displays the result: "Benign," "Malignant," or "Normal."

# Accuracy

The model achieves an accuracy of 91% on the test set.
