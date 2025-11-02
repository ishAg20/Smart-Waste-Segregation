# Proposed Model Implementation Guide

## 🎯 Overview

This guide covers the **proposed model** implementation for the Smart Waste Segregation system with significant improvements over the original MobileNetV2 baseline model.

## 🆕 Key Improvements

### 1. **Enhanced Architecture**
- **Base Model**: EfficientNetB0 (more powerful than MobileNetV2)
- **Attention Mechanism**: Squeeze-and-Excitation (SE) blocks for better feature focusing
- **Deeper Classification Head**: 256 → Dropout(0.4) → 128 → Dropout(0.3) → 3 classes

### 2. **Advanced Training Techniques**
- **Two-Stage Training**:
  - Stage 1: Train with frozen base (10 epochs)
  - Stage 2: Fine-tune with unfrozen base (15 epochs, lr=1e-5)
- **Label Smoothing**: 0.1 factor to reduce overconfidence
- **Focal Loss**: Handles class imbalance (gamma=2.0)
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning

### 3. **Data Enhancements**
- **70-30 Train-Test Split** (changed from 80-20)
- **Class Balancing**: RandomOverSampler for minority classes
- **Intelligent Augmentation**:
  - Rotation (±20°)
  - Zoom (±20%)
  - Brightness variation (80%-120%)
  - Horizontal flip
  - Width/height shifts

### 4. **Explainable AI**
- **Grad-CAM Visualization**: Shows what the model "sees"
- **Confidence Scores**: Per-class probability distribution
- **Misclassification Analysis**: Automatic detection and visualization

### 5. **Comprehensive Evaluation**
- Confusion matrix with detailed metrics
- Per-class precision, recall, F1-score
- Confidence distribution analysis
- Grad-CAM for misclassified samples

---

## 📁 New Files Created

```
Smart-Waste-Segregation/
├── model/
│   ├── proposed_model.py          # EfficientNetB0 + SE blocks architecture
│   ├── train_proposed.py          # Two-stage training with focal loss
│   └── evaluate_proposed.py       # Enhanced evaluation with Grad-CAM
├── utils/
│   └── gradcam.py                 # Grad-CAM visualization utilities
└── PROPOSED_MODEL_GUIDE.md        # This file
```

---

## 🚀 Usage Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**New dependencies added:**
- `focal-loss` - For handling class imbalance
- `tf-keras-vis` - For Grad-CAM visualization
- `pillow` - Image processing
- `imbalanced-learn` - Class balancing utilities

### Step 2: Preprocess Data (70-30 Split + Balancing)

```bash
python data_preprocessing/preprocess.py
```

**What happens:**
- Loads TrashNet dataset
- Applies class balancing (oversampling minority classes)
- Splits data into 70% train, 30% test
- Saves to `data_preprocessing/split_data.pkl`

**Expected output:**
```
Original class distribution: {0: 1200, 1: 800, 2: 600}
Balanced class distribution: {0: 1200, 1: 1200, 2: 1200}

Train set size: 2520 samples (70%)
Test set size: 1080 samples (30%)
```

### Step 3: Train Proposed Model

```bash
python model/train_proposed.py
```

**Training process:**

**Stage 1** (10 epochs):
- Base model (EfficientNetB0) frozen
- Only classification head trains
- Uses data augmentation
- Label smoothing (0.1)

**Stage 2** (15 epochs):
- Unfreezes last 50 layers of base model
- Fine-tunes with low learning rate (1e-5)
- Switches to Focal Loss (gamma=2.0)
- Continues augmentation

**Generated files:**
- `saved_models/proposed_model_best.h5` - Best model (highest val_accuracy)
- `saved_models/proposed_model_final.h5` - Final model after all epochs
- `saved_models/training_history_proposed.png` - Training curves
- `logs/proposed_model_TIMESTAMP/` - TensorBoard logs

### Step 4: Evaluate Model

```bash
python model/evaluate_proposed.py
```

**Evaluation outputs:**

1. **Console**: Classification report with precision, recall, F1-score
2. **Confusion Matrix**: `saved_models/confusion_matrix_proposed.png`
3. **Classification Report**: `saved_models/classification_report_proposed.txt`
4. **Confidence Analysis**: `saved_models/confidence_distribution_proposed.png`
5. **Grad-CAM Analysis**: `saved_models/misclassified_samples/gradcam_analysis.png`

### Step 5: Launch Streamlit App

```bash
streamlit run streamlit_app/app.py
```

**Features:**
- Model selection (Proposed vs Original)
- Real-time predictions
- Confidence score visualization
- Grad-CAM heatmap overlay
- Toggle visualizations on/off

---

## 📊 Model Comparison

| Feature | Original Model | Proposed Model |
|---------|---------------|----------------|
| **Base Architecture** | MobileNetV2 (frozen) | EfficientNetB0 (fine-tuned) |
| **Attention Mechanism** | ❌ None | ✅ SE blocks |
| **Training Strategy** | Single-stage | Two-stage |
| **Data Split** | 80-20 | 70-30 |
| **Class Balancing** | ❌ No | ✅ Yes (oversampling) |
| **Data Augmentation** | ❌ No | ✅ Yes (5 types) |
| **Label Smoothing** | ❌ No | ✅ Yes (0.1) |
| **Focal Loss** | ❌ No | ✅ Yes (gamma=2.0) |
| **Grad-CAM** | ❌ No | ✅ Yes |
| **Classification Head** | Simple (1 layer) | Deep (2 layers) |

---

## 🧠 Architecture Details

### Proposed Model Architecture

```
Input (224, 224, 3)
    ↓
EfficientNetB0 (ImageNet weights)
    ↓
SE Block (Squeeze-and-Excitation)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.4)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(3, activation='softmax')
    ↓
Output: [Biodegradable, Recyclable, Non-Recyclable]
```

### SE (Squeeze-and-Excitation) Block

```python
# Channel-wise attention mechanism
Squeeze: Global Average Pooling → [batch, channels]
Excitation: Dense(channels/16) → ReLU → Dense(channels) → Sigmoid
Scale: Multiply input feature maps by attention weights
```

**Purpose**: Helps model focus on important channels/features

---

## 🎨 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which parts of the image the model uses for its decision.

**Interpretation:**
- 🔴 **Red/Yellow**: High importance regions
- 🟢 **Green**: Moderate importance
- 🔵 **Blue**: Low importance

**Example use cases:**
1. **Debugging**: Check if model focuses on the object vs background
2. **Trust**: Understand model reasoning
3. **Improvement**: Identify training data issues

---

## 📈 Expected Performance Improvements

Based on the implemented techniques, you can expect:

✅ **+5-10% accuracy** improvement over baseline
✅ **Better generalization** (label smoothing, augmentation)
✅ **Reduced false positives** for similar classes (e.g., plastic cups)
✅ **Balanced performance** across all classes (focal loss)
✅ **More reliable predictions** (confidence calibration)

---

## 🔧 Customization Options

### Modify Training Parameters

Edit `model/train_proposed.py`:

```python
# Disable focal loss (use only label smoothing)
train_proposed_model(
    use_focal_loss=False,
    use_augmentation=True,
    two_stage_training=True
)

# Disable augmentation
train_proposed_model(
    use_focal_loss=True,
    use_augmentation=False,
    two_stage_training=True
)

# Single-stage training only
train_proposed_model(
    use_focal_loss=True,
    use_augmentation=True,
    two_stage_training=False
)
```

### Adjust Data Split Ratio

Edit `data_preprocessing/preprocess.py`:

```python
# Change from 70-30 to 60-40
return train_test_split(images, labels, test_size=0.4, random_state=42)
```

### Modify Augmentation Intensity

Edit `model/train_proposed.py`:

```python
datagen = ImageDataGenerator(
    rotation_range=30,        # Increase to 30°
    zoom_range=0.3,           # Increase zoom
    brightness_range=[0.7, 1.3],  # More brightness variation
    horizontal_flip=True,
    vertical_flip=True,       # Add vertical flip
    fill_mode='nearest'
)
```

---

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution**: Train the model first:
```bash
python model/train_proposed.py
```

### Issue: "Grad-CAM visualization failed"
**Solution**: The model needs at least one convolutional layer. EfficientNetB0 has many, so this should work. Check if model loaded correctly.

### Issue: Out of memory during training
**Solution**: Reduce batch size in `train_proposed.py`:
```python
batch_size=16  # Change from 32 to 16
```

### Issue: Training too slow
**Solution**:
1. Reduce number of epochs
2. Disable augmentation temporarily
3. Use GPU if available

---

## 📚 References

1. **EfficientNet**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
2. **SE Blocks**: [Hu et al., 2018](https://arxiv.org/abs/1709.01507)
3. **Focal Loss**: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
4. **Grad-CAM**: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)

---

## 🎓 Next Steps

1. **Hyperparameter Tuning**:
   - Grid search for learning rate
   - Experiment with different dropout rates
   - Try different SE block reduction ratios

2. **Alternative Base Models**:
   - ResNet50 (if you need higher capacity)
   - EfficientNetB1/B2 (better accuracy, slower)
   - MobileNetV3 (lightweight alternative)

3. **Ensemble Methods**:
   - Train multiple models with different seeds
   - Average predictions for better accuracy

4. **Deployment**:
   - Convert to TensorFlow Lite for mobile
   - Optimize with quantization
   - Deploy on edge devices (Raspberry Pi, etc.)

---

## ✅ Checklist

- [ ] Installed new dependencies
- [ ] Preprocessed data with 70-30 split
- [ ] Trained proposed model (2 stages)
- [ ] Evaluated model with Grad-CAM
- [ ] Tested Streamlit app
- [ ] Compared with original model
- [ ] Reviewed misclassified samples

---

**Happy Training! 🚀♻️**

For questions or issues, check the troubleshooting section or inspect the generated visualizations.
