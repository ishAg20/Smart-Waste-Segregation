"""
Model Comparison Script

Compares the original MobileNetV2 model with the proposed EfficientNetB0 model
on the test set and generates a side-by-side comparison report.

Usage:
    python compare_models.py
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model


def evaluate_model(model_path, X_test, y_test, model_name):
    """Evaluate a model on test data."""
    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_name}")
    print('=' * 70)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        return None

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, compile=False)

    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    cm = confusion_matrix(y_true, y_pred_classes)

    # Per-class accuracy
    categories = ["Biodegradable", "Recyclable", "Non-Recyclable"]
    class_accuracies = {}
    for i, category in enumerate(categories):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.sum(y_pred_classes[class_mask] == i) / np.sum(class_mask)
            class_accuracies[category] = class_acc

    # Confidence scores
    confidences = np.max(y_pred, axis=1)
    correct_mask = (y_pred_classes == y_true)
    avg_confidence = np.mean(confidences)
    correct_confidence = np.mean(confidences[correct_mask])
    incorrect_confidence = np.mean(confidences[~correct_mask]) if np.sum(~correct_mask) > 0 else 0

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_accuracies': class_accuracies,
        'avg_confidence': avg_confidence,
        'correct_confidence': correct_confidence,
        'incorrect_confidence': incorrect_confidence,
        'y_pred': y_pred,
        'y_pred_classes': y_pred_classes,
        'y_true': y_true
    }

    # Print summary
    print(f"\n✓ Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"✓ Average Confidence: {avg_confidence:.4f}")
    print(f"\nPer-class Accuracy:")
    for category, acc in class_accuracies.items():
        print(f"  {category:20s}: {acc * 100:.2f}%")

    return results


def plot_comparison(original_results, proposed_results):
    """Create comparison visualizations."""
    if original_results is None or proposed_results is None:
        print("\n⚠️ Cannot create comparison plots (one or both models missing)")
        return

    fig = plt.figure(figsize=(18, 10))

    categories = ["Biodegradable", "Recyclable", "Non-Recyclable"]

    # 1. Overall Accuracy Comparison
    plt.subplot(2, 3, 1)
    models = ['Original\n(MobileNetV2)', 'Proposed\n(EfficientNetB0)']
    accuracies = [
        original_results['accuracy'] * 100,
        proposed_results['accuracy'] * 100
    ]
    bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c'], alpha=0.8)
    plt.ylabel('Accuracy (%)')
    plt.title('Overall Accuracy Comparison', fontweight='bold')
    plt.ylim([0, 100])
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # 2. Per-Class Accuracy Comparison
    plt.subplot(2, 3, 2)
    x = np.arange(len(categories))
    width = 0.35
    original_class_acc = [original_results['class_accuracies'][cat] * 100 for cat in categories]
    proposed_class_acc = [proposed_results['class_accuracies'][cat] * 100 for cat in categories]

    plt.bar(x - width/2, original_class_acc, width, label='Original', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, proposed_class_acc, width, label='Proposed', color='#e74c3c', alpha=0.8)
    plt.xlabel('Category')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy Comparison', fontweight='bold')
    plt.xticks(x, categories, rotation=15, ha='right')
    plt.legend()
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3)

    # 3. Confidence Score Comparison
    plt.subplot(2, 3, 3)
    metrics = ['Avg\nConfidence', 'Correct\nConfidence', 'Incorrect\nConfidence']
    original_conf = [
        original_results['avg_confidence'],
        original_results['correct_confidence'],
        original_results['incorrect_confidence']
    ]
    proposed_conf = [
        proposed_results['avg_confidence'],
        proposed_results['correct_confidence'],
        proposed_results['incorrect_confidence']
    ]

    x = np.arange(len(metrics))
    plt.bar(x - width/2, original_conf, width, label='Original', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, proposed_conf, width, label='Proposed', color='#e74c3c', alpha=0.8)
    plt.ylabel('Confidence Score')
    plt.title('Prediction Confidence Comparison', fontweight='bold')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)

    # 4. Confusion Matrix - Original
    plt.subplot(2, 3, 4)
    cm_orig = original_results['confusion_matrix']
    im1 = plt.imshow(cm_orig, cmap='Blues', aspect='auto')
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.xticks(range(3), categories, rotation=15, ha='right')
    plt.yticks(range(3), categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Original', fontweight='bold')
    # Add text annotations
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm_orig[i, j]), ha='center', va='center',
                    color='white' if cm_orig[i, j] > cm_orig.max()/2 else 'black')

    # 5. Confusion Matrix - Proposed
    plt.subplot(2, 3, 5)
    cm_prop = proposed_results['confusion_matrix']
    im2 = plt.imshow(cm_prop, cmap='Reds', aspect='auto')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.xticks(range(3), categories, rotation=15, ha='right')
    plt.yticks(range(3), categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Proposed', fontweight='bold')
    # Add text annotations
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm_prop[i, j]), ha='center', va='center',
                    color='white' if cm_prop[i, j] > cm_prop.max()/2 else 'black')

    # 6. Improvement Summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    improvement = (proposed_results['accuracy'] - original_results['accuracy']) * 100
    improvement_sign = '+' if improvement > 0 else ''

    summary_text = f"""
    IMPROVEMENT SUMMARY

    Overall Accuracy:
    • Original: {original_results['accuracy'] * 100:.2f}%
    • Proposed: {proposed_results['accuracy'] * 100:.2f}%
    • Change: {improvement_sign}{improvement:.2f}%

    Per-Class Improvements:
    """

    for cat in categories:
        orig_acc = original_results['class_accuracies'][cat] * 100
        prop_acc = proposed_results['class_accuracies'][cat] * 100
        diff = prop_acc - orig_acc
        diff_sign = '+' if diff > 0 else ''
        summary_text += f"\n• {cat}:\n  {diff_sign}{diff:.2f}%"

    summary_text += f"""

    Confidence:
    • Original: {original_results['avg_confidence']:.4f}
    • Proposed: {proposed_results['avg_confidence']:.4f}
    """

    plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Model Comparison: Original vs Proposed', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    os.makedirs('saved_models', exist_ok=True)
    plt.savefig('saved_models/model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved to 'saved_models/model_comparison.png'")
    plt.close()


def main():
    print("=" * 70)
    print("MODEL COMPARISON SCRIPT")
    print("=" * 70)

    # Load test data
    print("\nLoading test data...")
    if not os.path.exists("data_preprocessing/split_data.pkl"):
        print("✗ Test data not found. Please run preprocessing first:")
        print("  python data_preprocessing/preprocess.py")
        return

    X_train, X_test, y_train, y_test = joblib.load("data_preprocessing/split_data.pkl")
    print(f"✓ Loaded {len(X_test)} test samples")

    # Evaluate original model
    original_results = evaluate_model(
        'saved_models/best_model.h5',
        X_test, y_test,
        'Original MobileNetV2'
    )

    # Evaluate proposed model
    proposed_model_path = 'saved_models/proposed_model_best.h5'
    if not os.path.exists(proposed_model_path):
        proposed_model_path = 'saved_models/proposed_model_final.h5'

    proposed_results = evaluate_model(
        proposed_model_path,
        X_test, y_test,
        'Proposed EfficientNetB0'
    )

    # Generate comparison plots
    print("\n" + "=" * 70)
    print("Generating comparison visualizations...")
    print("=" * 70)
    plot_comparison(original_results, proposed_results)

    # Print final summary
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    if original_results and proposed_results:
        improvement = (proposed_results['accuracy'] - original_results['accuracy']) * 100
        if improvement > 0:
            print(f"\n🎉 Proposed model is {improvement:.2f}% better than original!")
        elif improvement < 0:
            print(f"\n⚠️ Proposed model is {abs(improvement):.2f}% worse than original")
        else:
            print(f"\n📊 Both models have identical accuracy")

        print("\nGenerated files:")
        print("  - saved_models/model_comparison.png")
    else:
        print("\n⚠️ Could not complete comparison (missing model files)")
        print("\nTo train models:")
        print("  Original: python model/train.py")
        print("  Proposed: python model/train_proposed.py")


if __name__ == "__main__":
    main()
