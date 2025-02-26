import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Detect if running inside Docker and set model path accordingly
if os.path.exists("/.dockerenv"): 
    MODEL_PATH = '/app/models/mlp_model.h5'  # Use HDF5 format in Docker
else:
    MODEL_PATH = 'D:/My projects/Brain Tumor/mlp_model.h5'  # Local HDF5 format

# Load the model
try:
    model = load_model(MODEL_PATH)
except (OSError, ValueError) as e:
    st.error(f"Failed to load model. Error: {e}")

# Class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan to detect brain tumor types.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI Image')  

    # Preprocess the image
    img = img.convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Display prediction
    st.write(f"**Prediction:** {class_labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
