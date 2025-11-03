# Project Overview
Smart Waste Segregation is an end-to-end deep learning-based system for automatic waste image classification, sorting trash into Biodegradable, Recyclable, and Non-Recyclable categories. The repo features a data preprocessing pipeline, advanced model architecture (EfficientNetB0 + Squeeze-and-Excitation), robust training/evaluation scripts, and a full-featured Streamlit application for real-time inference and user interaction.​

# Core Workflow & Components
## 1. Data Preprocessing (data_preprocessing/preprocess.py)
- Scans the dataset folders, loads each image using OpenCV, resizes to 128×128, and scales pixel values.

- Converts raw class names to labels using utils/category_mapping.py: Biodegradable, Recyclable, Non-Recyclable

- Balances classes with RandomOverSampler.

- Splits the dataset into training/testing (70/30 split) and serializes to a single .pkl for downstream usage.​

## 2. Model Architecture (model/proposed_model.py)
- Backbone: EfficientNetB0 (pre-trained on ImageNet, initially frozen).

- Attention Module: Custom Squeeze-and-Excitation block for adaptive channel-wise attention.

- Classifier: Two dense layers (256 → 128, each followed by dropout), outputs 3-way softmax.

- Loss: Uses Categorical Crossentropy with label smoothing (smooths labels to reduce overfitting). Can optionally use Focal Loss during fine-tuning.

- Fine-tuning: Unfreezes deeper EfficientNet layers for second-stage training to improve accuracy and generalization.​

## 3. Training Pipeline (model/train_proposed.py)
- Augmentation: Rotation, shift, zoom, brightness change, and mirroring for robustness.
- Callbacks: Early stopping, best model checkpointing, learning rate scheduling, and TensorBoard logging.

- Two-Stage Training:
  
  Stage 1: Frozen base model, standard crossentropy.

  Stage 2: Unfreeze deeper layers, optionally switch to Focal Loss (handles class imbalance).

  Final Evaluation: Reports and saves accuracy, loss, and training curves.​

## 4. Evaluation & Explainability (model/evaluate_proposed.py, utils/gradcam.py)
- Evaluates on test data, prints confusion matrix, precision, recall, and F1.

- Visualizations saved as images for further inspection.​

## 5. Inference Pipeline (inference/predict.py)
- Loads a saved trained model.

- Accepts an image path, pre-processes the image, runs model prediction, and returns the class label.​

## 6. Streamlit App (streamlit_app/app.py)
- UI: User-friendly interface for dragging and dropping waste images.

- Prediction: Shows the uploaded image, its predicted category, a description of the category, and model confidence.

- Accessibility: Model details, usage guide, and example categories are all built into the app.

# Quick Start
## Installation

### Clone the repo and enter the directory
```
git clone https://github.com/ishAg20/Smart-Waste-Segregation.git
cd Smart-Waste-Segregation
```

### Create & activate a virtual environment
```
python -m venv venv
venv\Scripts\activate        # Windows
# OR
source venv/bin/activate     # macOS/Linux
```

### Install all dependencies
```
pip install -r requirements.txt
```

## Data Preparation
Organize your dataset in TrashNet-like format with subfolders for each class (paper, cardboard, metal, glass, trash, plastic).
Run:
```
python data_preprocessing/preprocess.py
```

## Training
```
python model/train_proposed.py
```
Saves best and final model checkpoints in saved_models/.

## Evaluation
```
python model/evaluate_proposed.py
```
Outputs test accuracy, confusion matrix, full classification report, and Grad-CAM analyses.

## Streamlit Inference Demo
```
streamlit run streamlit_app/app.py
```
Visit the local Streamlit link provided and upload waste images for instant results and explanations.

# Key Features
State-of-the-art Deep Learning: EfficientNet+SE

Robust Training: Augmentation, focal loss, label smoothing, two-stage fine-tuning

User-facing Demo: Streamlit app for hands-on use and validation by non-technical users

# File & Folder Summary
- data_preprocessing/preprocess.py: Loads, balances, splits, and serializes dataset.​

- model/proposed_model.py: EfficientNetB0 architecture enhanced with Squeeze-and-Excitation.​

- model/train_proposed.py: Training pipeline (augmentation, two-stage learning).​

- model/evaluate_proposed.py: Full model evaluation.​

- inference/predict.py: Simple CLI image predictor.​

- streamlit_app/app.py: Streamlit UI, model selector, prediction.

- utils/category_mapping.py: Category mapping for harmonized labels.
