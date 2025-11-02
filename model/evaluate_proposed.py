import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score
)
from tensorflow.keras.models import load_model
import os
import cv2
from utils.gradcam import visualize_gradcam


def evaluate_proposed_model(model_path='saved_models/proposed_model_best.h5'):
    """
    Comprehensive evaluation of the proposed model with:
    - Confusion matrix
    - Per-class metrics (precision, recall, F1-score)
    - Overall accuracy
    - Grad-CAM visualization for misclassified samples
    """
    print("=" * 70)
    print("PROPOSED MODEL EVALUATION - Smart Waste Segregation")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading trained model...")
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        print("Please train the model first using: python model/train_proposed.py")
        return

    model = load_model(model_path, compile=False)
    print(f"✓ Model loaded from {model_path}")

    # Load test data
    print("\n[2/5] Loading test data...")
    X_train, X_test, y_train, y_test = joblib.load("data_preprocessing/split_data.pkl")
    print(f"✓ Test set: {X_test.shape[0]} samples")

    # Make predictions
    print("\n[3/5] Making predictions on test set...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"✓ Test Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    print("\n[4/5] Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred_classes)
    categories = ["Biodegradable", "Recyclable", "Non-Recyclable"]

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Proposed Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('saved_models/confusion_matrix_proposed.png', dpi=150)
    print("✓ Confusion matrix saved to 'saved_models/confusion_matrix_proposed.png'")
    plt.close()

    # Classification Report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(
        y_true, y_pred_classes,
        target_names=categories,
        digits=4
    )
    print(report)

    # Save report
    with open('saved_models/classification_report_proposed.txt', 'w') as f:
        f.write("PROPOSED MODEL - Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print("✓ Report saved to 'saved_models/classification_report_proposed.txt'")

    # Per-class accuracy
    print("\n" + "=" * 70)
    print("PER-CLASS ACCURACY")
    print("=" * 70)
    for i, category in enumerate(categories):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.sum(y_pred_classes[class_mask] == i) / np.sum(class_mask)
            print(f"{category:20s}: {class_acc * 100:.2f}%")

    # Analyze misclassifications
    print("\n[5/5] Analyzing misclassified samples...")
    misclassified_indices = np.where(y_pred_classes != y_true)[0]
    print(f"✓ Found {len(misclassified_indices)} misclassified samples")

    if len(misclassified_indices) > 0:
        # Create directory for misclassified samples
        os.makedirs('saved_models/misclassified_samples', exist_ok=True)

        # Save first 10 misclassified samples with Grad-CAM
        num_to_save = min(10, len(misclassified_indices))
        print(f"\nGenerating Grad-CAM visualizations for {num_to_save} misclassified samples...")

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, idx in enumerate(misclassified_indices[:num_to_save]):
            # Get image and predictions
            img = X_test[idx]
            true_label = categories[y_true[idx]]
            pred_label = categories[y_pred_classes[idx]]
            confidence = y_pred[idx][y_pred_classes[idx]]

            # Save temporary image for Grad-CAM
            temp_path = f'saved_models/misclassified_samples/temp_{i}.jpg'
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.imwrite(temp_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

            # Generate Grad-CAM (with error handling)
            try:
                gradcam_img, _, _ = visualize_gradcam(
                    temp_path,
                    model,
                    last_conv_layer_name=None  # Auto-detect
                )
                axes[i].imshow(gradcam_img)
            except Exception as e:
                print(f"  Warning: Could not generate Grad-CAM for sample {i}: {e}")
                axes[i].imshow(img)

            axes[i].set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence:.2%})",
                fontsize=10
            )
            axes[i].axis('off')

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        plt.tight_layout()
        plt.savefig('saved_models/misclassified_samples/gradcam_analysis.png', dpi=150)
        print("✓ Grad-CAM analysis saved to 'saved_models/misclassified_samples/gradcam_analysis.png'")
        plt.close()

    # Confidence distribution
    print("\n[Bonus] Analyzing prediction confidence...")
    plt.figure(figsize=(12, 5))

    # Overall confidence distribution
    plt.subplot(1, 2, 1)
    confidences = np.max(y_pred, axis=1)
    plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Overall Prediction Confidence Distribution')
    plt.axvline(np.mean(confidences), color='r', linestyle='--',
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confidence by correctness
    plt.subplot(1, 2, 2)
    correct_mask = (y_pred_classes == y_true)
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', edgecolor='black')
    plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('saved_models/confidence_distribution_proposed.png', dpi=150)
    print("✓ Confidence analysis saved to 'saved_models/confidence_distribution_proposed.png'")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Test Accuracy:          {accuracy * 100:.2f}%")
    print(f"Total Samples:          {len(y_test)}")
    print(f"Correct Predictions:    {np.sum(correct_mask)}")
    print(f"Incorrect Predictions:  {np.sum(~correct_mask)}")
    print(f"Average Confidence:     {np.mean(confidences):.4f}")
    print(f"Correct Avg Conf:       {np.mean(correct_conf):.4f}")
    if len(incorrect_conf) > 0:
        print(f"Incorrect Avg Conf:     {np.mean(incorrect_conf):.4f}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - saved_models/confusion_matrix_proposed.png")
    print("  - saved_models/classification_report_proposed.txt")
    print("  - saved_models/confidence_distribution_proposed.png")
    print("  - saved_models/misclassified_samples/gradcam_analysis.png")


if __name__ == "__main__":
    # Evaluate the proposed model
    evaluate_proposed_model()
