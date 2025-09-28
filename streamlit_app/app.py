import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from inference.predict import predict_image

st.title("♻️ Smart Waste Classifier")
st.write("Upload a waste image to classify it as biodegradable, recyclable, or non-recyclable.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)
    label = predict_image("temp.jpg")
    st.success(f"Predicted Waste Type: **{label}**")
