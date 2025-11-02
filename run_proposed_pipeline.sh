#!/bin/bash

# Smart Waste Segregation - Proposed Model Pipeline
# This script runs the complete pipeline for the proposed model

set -e  # Exit on error

echo "======================================================================"
echo "  Smart Waste Segregation - Proposed Model Training Pipeline"
echo "======================================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP $1/5]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.7+."
    exit 1
fi

print_success "Python found: $(python --version)"
echo ""

# Step 1: Check dependencies
print_step "1" "Checking dependencies..."
if python -c "import tensorflow, streamlit, cv2, sklearn" 2>/dev/null; then
    print_success "All basic dependencies installed"
else
    print_warning "Some dependencies missing. Installing..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
fi
echo ""

# Step 2: Check dataset
print_step "2" "Checking dataset..."
if [ -d "dataset/TrashNet" ]; then
    print_success "TrashNet dataset found"
else
    print_error "TrashNet dataset not found at dataset/TrashNet/"
    echo ""
    echo "Please download the dataset:"
    echo "1. Go to https://github.com/garythung/trashnet"
    echo "2. Download and extract to dataset/TrashNet/"
    echo "3. Re-run this script"
    exit 1
fi
echo ""

# Step 3: Preprocess data
print_step "3" "Preprocessing data (70-30 split with class balancing)..."
if python data_preprocessing/preprocess.py; then
    print_success "Data preprocessing complete"
    print_success "Data saved to data_preprocessing/split_data.pkl"
else
    print_error "Preprocessing failed"
    exit 1
fi
echo ""

# Step 4: Train proposed model
print_step "4" "Training proposed model (this may take 30-40 minutes)..."
echo "Training stages:"
echo "  - Stage 1: Frozen base (10 epochs)"
echo "  - Stage 2: Fine-tuning (15 epochs)"
echo ""

if python model/train_proposed.py; then
    print_success "Model training complete"
    print_success "Best model saved to saved_models/proposed_model_best.h5"
    print_success "Final model saved to saved_models/proposed_model_final.h5"
else
    print_error "Training failed"
    exit 1
fi
echo ""

# Step 5: Evaluate model
print_step "5" "Evaluating proposed model..."
if python model/evaluate_proposed.py; then
    print_success "Evaluation complete"
    echo ""
    echo "Generated files:"
    echo "  - saved_models/confusion_matrix_proposed.png"
    echo "  - saved_models/classification_report_proposed.txt"
    echo "  - saved_models/confidence_distribution_proposed.png"
    echo "  - saved_models/misclassified_samples/gradcam_analysis.png"
else
    print_error "Evaluation failed"
    exit 1
fi
echo ""

# Completion message
echo "======================================================================"
echo -e "${GREEN}✓ PIPELINE COMPLETE!${NC}"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. View training history:"
echo "   → open saved_models/training_history_proposed.png"
echo ""
echo "2. Review evaluation results:"
echo "   → open saved_models/confusion_matrix_proposed.png"
echo "   → open saved_models/classification_report_proposed.txt"
echo ""
echo "3. Compare with original model (optional):"
echo "   → python compare_models.py"
echo ""
echo "4. Launch web interface:"
echo "   → streamlit run streamlit_app/app.py"
echo ""
echo "5. View TensorBoard logs (optional):"
echo "   → tensorboard --logdir=logs/"
echo ""
echo "======================================================================"
echo -e "${BLUE}Documentation:${NC}"
echo "  - PROPOSED_MODEL_GUIDE.md - Comprehensive guide"
echo "  - IMPLEMENTATION_SUMMARY.md - Implementation details"
echo "  - README.md - Quick start"
echo "======================================================================"
