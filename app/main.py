import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st


# print("NumPy version:", np.__version__)
# print("Streamlit version:", st.__version__)
# print("TensorFlow version:", tf.__version__)
# plant_disease_prediction_model.h5
model_path = "Models/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

with open("Results/class_indices.json", "r") as f:
    class_indices = json.load(f)

# check input shape
input_height, input_width = model.input_shape[1:3]

def load_and_preprocess_image(image_path, target_size=(input_height, input_width)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') / 255.0
    return image_array

def prediction_image(model, image_path, class_indices):
    processed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(processed_image)
    prediction_class_index = int(np.argmax(predictions, axis=-1)[0])
    prediction_class_name = class_indices[str(prediction_class_index)]
    return prediction_class_name

st.title("Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((150,150))
        st.image(resized_image)

    with col2:
        if st.button('Classify'):
            prediction = prediction_image(model, uploaded_image, class_indices)
            st.success(f'Prediction: {prediction}')
