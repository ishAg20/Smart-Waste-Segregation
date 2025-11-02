# Quick Reference Card

## 🚀 One-Command Pipeline

```bash
# Run complete pipeline (preprocess → train → evaluate)
./run_proposed_pipeline.sh
```

---

## 📝 Individual Commands

### Data Preprocessing
```bash
# 70-30 split with class balancing
python data_preprocessing/preprocess.py
```

### Training
```bash
# Train proposed model (30-40 min)
python model/train_proposed.py

# Train original model (10-15 min)
python model/train.py
```

### Evaluation
```bash
# Evaluate proposed model
python model/evaluate_proposed.py

# Evaluate original model
python model/evaluate.py

# Compare both models
python compare_models.py
```

### Web Interface
```bash
# Launch Streamlit app
streamlit run streamlit_app/app.py
```

### TensorBoard (Optional)
```bash
# View training logs
tensorboard --logdir=logs/
```

---

## 📁 Important Files

### Models
- `saved_models/best_model.h5` - Original MobileNetV2
- `saved_models/proposed_model_best.h5` - Proposed EfficientNetB0 (best)
- `saved_models/proposed_model_final.h5` - Proposed (final)

### Data
- `data_preprocessing/split_data.pkl` - Preprocessed train/test data

### Results
- `saved_models/confusion_matrix_proposed.png` - Confusion matrix
- `saved_models/classification_report_proposed.txt` - Metrics
- `saved_models/training_history_proposed.png` - Training curves
- `saved_models/confidence_distribution_proposed.png` - Confidence analysis
- `saved_models/misclassified_samples/gradcam_analysis.png` - Error analysis

---

## 🎯 Model Comparison

| Feature | Original | Proposed |
|---------|----------|----------|
| **Base** | MobileNetV2 | EfficientNetB0 |
| **SE Blocks** | ❌ | ✅ |
| **Training** | 1 stage | 2 stages |
| **Split** | 80-20 | 70-30 |
| **Balancing** | ❌ | ✅ |
| **Augmentation** | ❌ | ✅ |
| **Focal Loss** | ❌ | ✅ |
| **Grad-CAM** | ❌ | ✅ |

---

## 💡 Quick Tips

### Speed Up Training
```bash
# Reduce epochs in train_proposed.py
# Stage 1: 10 → 5 epochs
# Stage 2: 15 → 10 epochs
```

### Reduce Memory Usage
```bash
# Reduce batch size in train_proposed.py
batch_size=16  # Change from 32 to 16
```

### Skip Augmentation (faster, less accurate)
```python
# In train_proposed.py
train_proposed_model(
    use_augmentation=False,  # Disable
    use_focal_loss=True,
    two_stage_training=True
)
```

### Disable Grad-CAM in Streamlit
```python
# In app.py sidebar
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=False)
```

---

## 🔍 Troubleshooting

### "Model not found"
```bash
# Train the model first
python model/train_proposed.py
```

### "Dataset not found"
```bash
# Download TrashNet dataset
# Place in: dataset/TrashNet/
```

### "Out of memory"
```bash
# Reduce batch size
# Or close other applications
# Or use CPU (slower)
```

### "Import error"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

## 📖 Documentation

- **Quick Start**: [README.md](README.md)
- **Detailed Guide**: [PROPOSED_MODEL_GUIDE.md](PROPOSED_MODEL_GUIDE.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## 🎨 Streamlit App Features

### Sidebar Settings
- Model selection (Original/Proposed)
- Toggle Grad-CAM visualization
- Toggle confidence scores

### Main Features
- Upload image (JPG, JPEG, PNG)
- Real-time prediction
- Confidence bars for all classes
- Grad-CAM heatmap overlay
- Model architecture info

---

## 📊 Expected Results

### Original Model
- Accuracy: **75-85%**
- Training: **10-15 min**
- Model size: **~14MB**

### Proposed Model
- Accuracy: **85-92%** (expected)
- Training: **30-40 min**
- Model size: **~25MB**
- + Grad-CAM
- + Better class balance

---

## 🛠️ Customization

### Change Split Ratio
```python
# In preprocess.py
train_test_split(images, labels, test_size=0.4)  # 60-40
```

### Modify Augmentation
```python
# In train_proposed.py
datagen = ImageDataGenerator(
    rotation_range=30,      # Increase
    zoom_range=0.3,         # Increase
    brightness_range=[0.7, 1.3]  # Wider range
)
```

### Try Different Base Model
```python
# In proposed_model.py
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

---

## ⚡ Keyboard Shortcuts

### In Streamlit App
- `r` - Rerun app
- `c` - Clear cache
- `Ctrl+C` - Stop server

### In Terminal
- `Ctrl+C` - Stop training/script
- `Ctrl+Z` - Suspend process
- `fg` - Resume process

---

## 📞 Getting Help

1. Check error message
2. Review documentation
3. Check generated visualizations
4. Compare original vs proposed
5. Review training logs

---

**Need more details? See [PROPOSED_MODEL_GUIDE.md](PROPOSED_MODEL_GUIDE.md)**
