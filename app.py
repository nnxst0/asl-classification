import streamlit as st
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title 
st.title('American Sign Language Classification')

# Set Header 
st.header('Please upload a picture')

# Load Model 
model = tf.keras.models.load_model('asl_model_fold0.h5')

# Display image & Prediction  
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    class_name = ['a', 'b', 'c', 'd']

    if st.button('Prediction'):
        # Prediction class
        label, probli = pred_class(image, model)

        st.write("## Prediction Result")
        st.write(f"## <span style='color:blue'>{label} : {probli[0][np.argmax(probli[0])]*100:.2f}%</span>",
                 unsafe_allow_html=True)
