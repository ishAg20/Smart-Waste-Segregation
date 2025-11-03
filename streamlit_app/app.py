import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from utils.gradcam import visualize_gradcam, get_last_conv_layer_name
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 🧩 Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Classifier")
st.markdown("""
Upload a waste image to classify it as **Biodegradable**, **Recyclable**, or **Non-Recyclable**.  
This app uses deep learning with explainable AI (Grad-CAM) to visualize what the model focuses on.
""")

# ---------------------------------------------------------
# ⚙️ Sidebar - Model Selection
# ---------------------------------------------------------
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Proposed Model (EfficientNetB0)", "Original Model (MobileNetV2)"]
)

# ---------------------------------------------------------
# 🧠 Load Selected Model
# ---------------------------------------------------------
@st.cache_resource
def load_selected_model(choice):
    if choice == "Proposed Model (EfficientNetB0)":
        model_path = "saved_models/proposed_model_best.h5"
        if not os.path.exists(model_path):
            model_path = "saved_models/proposed_model_final.h5"
    else:
        model_path = "saved_models/best_model.h5"

    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
        input_shape = model.input_shape[1:3]  # dynamically detect
        return model, model_path, input_shape
    else:
        return None, None, None

model, model_path, target_size = load_selected_model(model_choice)

if model is None:
    st.error("⚠️ Model not found. Please train the model first.")
    st.info("""
    **To train the proposed model:**
    1. `python data_preprocessing/preprocess.py`
    2. `python model/train_proposed.py`

    **To train the original model:**
    1. `python data_preprocessing/preprocess.py`
    2. `python model/train.py`
    """)
    st.stop()

st.sidebar.success(f"✓ Model loaded: {os.path.basename(model_path)}")
st.sidebar.info(f"📏 Model input size: {target_size}")

# ---------------------------------------------------------
# 📤 Image Upload Section
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # -----------------------------------------------------
    # 🖼️ Preprocess Image
    # -----------------------------------------------------
    img = cv2.imread(temp_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)  # ✅ dynamic resizing
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------------------------------
    # 🎯 Make Prediction
    # -----------------------------------------------------
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_class])

    categories = ["Biodegradable", "Recyclable", "Non-Recyclable"]
    predicted_label = categories[pred_class]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(img_rgb, use_column_width=True)

    with col2:
        st.subheader("🎯 Prediction Results")

        if predicted_label == "Biodegradable":
            st.success(f"### {predicted_label}")
            st.info("♻️ This waste can decompose naturally.")
        elif predicted_label == "Recyclable":
            st.info(f"### {predicted_label}")
            st.info("🔄 This waste can be recycled.")
        else:
            st.warning(f"### {predicted_label}")
            st.info("🗑️ This waste cannot be recycled or composted.")


    # -----------------------------------------------------
    # 📘 About Section
    # -----------------------------------------------------
    st.markdown("---")
    with st.expander("ℹ️ About the Model"):
        if "Proposed" in model_choice:
            st.markdown("""
            **Proposed Model Architecture:**
            - Base: EfficientNetB0 (pre-trained on ImageNet)
            - SE (Squeeze-and-Excitation) blocks for attention
            - Dense layers: 256 → Dropout(0.4) → 128 → Dropout(0.3) → 3
            - Training: Two-stage (frozen → fine-tuned)
            - Loss: Categorical Crossentropy with Label Smoothing + Focal Loss
            - Data: 70-30 train-test split with class balancing
            - Augmentation: Rotation, zoom, brightness, flip
            """)
        else:
            st.markdown("""
            **Original Model Architecture:**
            - Base: MobileNetV2 (pre-trained on ImageNet, frozen)
            - Dense layers: GlobalAveragePooling → Dropout(0.3) → 3
            - Training: Single-stage with frozen base
            - Loss: Categorical Crossentropy
            - Data: 80-20 train-test split
            """)

    # Cleanup temporary image
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    # -----------------------------------------------------
    # 🧾 Info & Examples
    # -----------------------------------------------------
    st.info("👆 Please upload an image to get started!")
    st.markdown("---")
    st.markdown("### 📋 Example Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌱 Biodegradable**")
        st.markdown("- Paper\n- Cardboard\n- Food waste\n- Organic materials")

    with col2:
        st.markdown("**♻️ Recyclable**")
        st.markdown("- Glass bottles\n- Metal cans\n- Aluminum foil")

    with col3:
        st.markdown("**🗑️ Non-Recyclable**")
        st.markdown("- Plastic bags\n- Styrofoam\n- Mixed materials")
