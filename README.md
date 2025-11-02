# Smart-Waste-Segregation

This project implements a **Smart Waste Segregation and Management System** using deep learning and computer vision. It classifies waste images into **three broad categories** — **Biodegradable**, **Recyclable**, and **Non-Recyclable** — using a Convolutional Neural Network (CNN) built on top of **MobileNetV2**. The system also features a **Streamlit web interface** that allows users to upload images and get instant classification results.

## Key Features

- Uses the [TrashNet Dataset](https://github.com/garythung/trashnet) with mapped classes
- **Two model options**:
  - **Original**: MobileNetV2 + Transfer Learning
  - **Proposed**: EfficientNetB0 + SE blocks + Advanced training techniques
- Custom preprocessing and mapping to 3 broader waste categories
- **70-30 train-test split** with class balancing
- **Advanced training**: Two-stage training, focal loss, label smoothing, data augmentation
- Model evaluation with **accuracy metrics**, **confusion matrix**, and **Grad-CAM visualization**
- Streamlit-based **web interface** with explainable AI features
- Modular codebase for easy upgrades and experimentation

## Setup Instructions

### 1. Clone the Repository

### 2. Create two folders - 
- dataset/TrashNet/
- saved_models

### 3. Download the TrashNet dataset:

Link: https://github.com/garythung/trashnet

Extract and place it inside:

dataset/TrashNet/

Ensure the structure looks like:

dataset/TrashNet/

├── glass/

├── metal/

├── paper/

├── cardboard/

├── plastic/

└── trash/

### 4. Install Dependencies

Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate
```

### 5. Install required packages:

```
pip install -r requirements.txt
```

### 6. Running the Project

1. Preprocess the Data

```
python data_preprocessing/preprocess.py
```

OR

```
python -m data_preprocessing.preprocess
```

This will:

- Resize images to 224x224
- Normalize pixel values
- Map original labels into 3 categories
- Split into train/test sets
- Save the split data as a .pkl file

2. Train the Model

```
python model/train.py
```

This will:

- Load preprocessed data
- Build and train a MobileNetV2-based CNN
- Save the best model as saved_models/best_model.h5

3. Evaluate the Model

```
python model/evaluate.py
```

This will:

- Load the best model
- Evaluate it on the test set
- Display test accuracy and confusion matrix

4. Launch the Web Interface

```
streamlit run streamlit_app/app.py
```

A browser window will open.

Upload an image, and you'll instantly see the predicted category!

---

## 🚀 Using the Proposed Model (Recommended)

For **better accuracy** and **explainable AI features**, use the proposed model:

### Quick Start

```bash
# 1. Preprocess with 70-30 split and class balancing
python data_preprocessing/preprocess.py

# 2. Train proposed model (two-stage training)
python model/train_proposed.py

# 3. Evaluate with detailed metrics and Grad-CAM
python model/evaluate_proposed.py

# 4. Launch Streamlit app (supports both models)
streamlit run streamlit_app/app.py
```

### What's Different?

✅ **EfficientNetB0** base (better than MobileNetV2)
✅ **SE (Squeeze-and-Excitation) blocks** for attention
✅ **70-30 train-test split** (changed from 80-20)
✅ **Class balancing** with oversampling
✅ **Two-stage training** (frozen → fine-tuning)
✅ **Focal loss** for handling imbalanced classes
✅ **Label smoothing** for better generalization
✅ **Data augmentation** (rotation, zoom, brightness, flip)
✅ **Grad-CAM visualization** for explainability

📖 **For detailed documentation, see [PROPOSED_MODEL_GUIDE.md](PROPOSED_MODEL_GUIDE.md)**

---

## 📊 Model Comparison

| Feature | Original | Proposed |
|---------|----------|----------|
| Base Model | MobileNetV2 | EfficientNetB0 |
| Training | Single-stage | Two-stage |
| Data Split | 80-20 | 70-30 |
| Augmentation | ❌ | ✅ |
| Class Balancing | ❌ | ✅ |
| Focal Loss | ❌ | ✅ |
| Grad-CAM | ❌ | ✅ |

