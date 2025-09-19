import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("mnist_cnn.h5")
class_names = [str(i) for i in range(10)]

st.title("🖊️ MNIST Digit Classifier")
st.write("Upload a handwritten digit image (28x28 pixels recommended).")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Convert to grayscale and preprocess
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image.resize((28,28)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=(0,-1))

    # Predict
    pred = model.predict(img_array)
    label = class_names[np.argmax(pred)]

    st.image(image, caption=f"Prediction: {label}", width=150)
