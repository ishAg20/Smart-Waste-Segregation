import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from utils.category_mapping import map_label

# Reduced size for memory efficiency
IMG_SIZE = 128 

def load_data(dataset_path, balance_classes=False):
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
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0   # use float32
                images.append(img)
                labels.append(map_label(class_name))
            except Exception as e:
                print(f"Skipped {img_file}: {e}")
                continue

    images = np.array(images, dtype='float32')
    labels_array = np.array(labels, dtype='int32')

    print("Original class distribution:", Counter(labels_array))

    # Balance classes
    if balance_classes:
        n_samples = images.shape[0]
        images_flat = images.reshape(n_samples, -1)

        # Use float32 for reduced memory footprint
        images_flat = images_flat.astype('float32')

        ros = RandomOverSampler(random_state=42)
        try:
            images_balanced, labels_balanced = ros.fit_resample(images_flat, labels_array)
            images = images_balanced.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            labels_array = labels_balanced
            print("Balanced class distribution:", Counter(labels_array))
        except MemoryError:
            print("⚠️  MemoryError: Dataset too large to balance in RAM.")
            print("Tip: Use smaller IMG_SIZE or rely on data augmentation instead.")
            # Fall back to unbalanced dataset
            pass

    labels = to_categorical(labels_array, num_classes=3)

    return train_test_split(images, labels, test_size=0.3, random_state=42)

if __name__ == "__main__":
    from joblib import dump

    X_train, X_test, y_train, y_test = load_data("dataset/TrashNet", balance_classes=True)

    print(f"\nTrain set size: {len(X_train)} samples (70%)")
    print(f"Test set size: {len(X_test)} samples (30%)")

    dump((X_train, X_test, y_train, y_test), "data_preprocessing/split_data.pkl")
    print("\nData saved to data_preprocessing/split_data.pkl")
