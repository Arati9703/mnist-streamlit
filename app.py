import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load model
model = load_model("mnist_cnn.h5")

st.title("🖊️ MNIST Digit Recognizer")
st.write("Draw a digit (0–9) in the box below and let the model predict!")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))  # drop alpha
    img = img.convert("L").resize((28, 28))  # grayscale + resize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    pred = model.predict(img_array)
    st.write(f"**Prediction:** {np.argmax(pred)}")
