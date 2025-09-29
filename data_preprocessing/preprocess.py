import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.category_mapping import map_label

IMG_SIZE = 224

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

def load_data(dataset_path):
    images, labels = [], []
    class_names = os.listdir(dataset_path)

    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                images.append(img)
                labels.append(map_label(class_name))
            except:
                continue

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=3)
    return train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

if __name__ == "__main__":
    from joblib import dump
    X_train, X_test, y_train, y_test = load_data("dataset/TrashNet")
    dump((X_train, X_test, y_train, y_test), "data_preprocessing/split_data.pkl")
