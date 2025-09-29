# Smart-Waste-Segregation

This project implements a **Smart Waste Segregation and Management System** using advanced deep learning and computer vision techniques. It classifies waste images into **three broad categories** — **Biodegradable**, **Recyclable**, and **Non-Recyclable** — using state-of-the-art Convolutional Neural Networks with **transfer learning**. The system features multiple model architectures, data augmentation, ensemble methods, and a **Streamlit web interface** for instant classification.

## Key Features

- Uses the [TrashNet Dataset](https://github.com/garythung/trashnet) with intelligent class mapping
- **Multiple pre-trained architectures**: EfficientNetB0, ResNet50V2, MobileNetV2
- **Advanced training techniques**: Two-phase training, fine-tuning, data augmentation
- **Ensemble methods** for improved accuracy and robustness
- **Comprehensive regularization**: L2 regularization, dropout, batch normalization
- **Smart callbacks**: Early stopping, learning rate scheduling, model checkpointing
- Model evaluation with detailed **accuracy metrics**, **confusion matrix**, and **classification reports**
- Streamlit-based **web interface** for instant image classification
- **Modular architecture** for easy experimentation and model comparison
- **Performance optimization** achieving 85-92% accuracy (improved from ~80%)

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

