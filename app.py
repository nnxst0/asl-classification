import streamlit as st
from PIL import Image
from prediction import pred_class
import torch
import numpy as np

# Set title 
st.title('American Sign Language Classification')

# Set Header 
st.header('Please upload a picture')

# Load Model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check the model
try:
    model = torch.load('asl_model_fold0.pt', map_location=device)
    st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Display image & Prediction  
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    class_names = ['a', 'b', 'c', 'd']

    if st.button('Prediction'):
        # Prediction class
        try:
            label, probli = pred_class(model, image, class_names)
            st.write("## Prediction Result")
            max_index = np.argmax(probli[0])

            for i in range(len(class_names)):
                color = "blue" if i == max_index else None
                st.write(f"## <span style='color:{color}'>{class_names[i]} : {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)
        except Exception as e:
            st.write(f"Error during prediction: {e}")
