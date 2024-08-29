import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow.keras

# ฟังก์ชั่นในการทำ Classification
def classify_image(image, model_path):
    model = tensorflow.keras.models.load_model(model_path)
    data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
    image = ImageOps.fit(image, (64, 64), Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return np.argmax(prediction), prediction

st.title("ASL Classification")
st.header("Upload a hand sign image to classify")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    label, prob = classify_image(image, 'asl_model.h5')
    
    # แสดงผลลัพธ์
    classes = ['A', 'B', 'C', 'D']
    st.write(f"Prediction: {classes[label]} with confidence {prob[0][label]:.2%}")
