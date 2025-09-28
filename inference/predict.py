import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("saved_models/best_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    categories = ["Biodegradable", "Recyclable", "Non-Recyclable"]
    return categories[class_idx]
