import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from ensemble_model import EnsembleModel
from train import train_improved_model

def compare_models():
    """Compare different model configurations"""

    # Load test data
    X_train, X_test, y_train, y_test = joblib.load("data_preprocessing/split_data.pkl")

    results = {}

    # Test 1: Original MobileNet
    print("Testing original MobileNet...")
    try:
        original_model = load_model("saved_models/best_model.h5")
        original_pred = original_model.predict(X_test)
        original_acc = np.mean(np.argmax(original_pred, axis=1) == np.argmax(y_test, axis=1))
        results['Original MobileNet'] = original_acc
        print(f"Original MobileNet accuracy: {original_acc:.4f}")
    except:
        print("Original model not found, skipping...")

    # Test 2: Improved EfficientNet
    print("Testing improved EfficientNet...")
    model_eff, _ = train_improved_model('efficientnet', use_augmentation=True, fine_tune_phase=False)
    eff_pred = model_eff.predict(X_test)
    eff_acc = np.mean(np.argmax(eff_pred, axis=1) == np.argmax(y_test, axis=1))
    results['EfficientNet + Augmentation'] = eff_acc
    print(f"EfficientNet accuracy: {eff_acc:.4f}")

    # Test 3: EfficientNet with fine-tuning
    print("Testing EfficientNet with fine-tuning...")
    model_eff_ft, _ = train_improved_model('efficientnet', use_augmentation=True, fine_tune_phase=True)
    eff_ft_pred = model_eff_ft.predict(X_test)
    eff_ft_acc = np.mean(np.argmax(eff_ft_pred, axis=1) == np.argmax(y_test, axis=1))
    results['EfficientNet + Fine-tuning'] = eff_ft_acc
    print(f"EfficientNet with fine-tuning accuracy: {eff_ft_acc:.4f}")

    # Test 4: ResNet
    print("Testing ResNet...")
    model_resnet, _ = train_improved_model('resnet', use_augmentation=True, fine_tune_phase=False)
    resnet_pred = model_resnet.predict(X_test)
    resnet_acc = np.mean(np.argmax(resnet_pred, axis=1) == np.argmax(y_test, axis=1))
    results['ResNet + Augmentation'] = resnet_acc
    print(f"ResNet accuracy: {resnet_acc:.4f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = list(results.values())

    plt.bar(models, accuracies, alpha=0.7)
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Detailed evaluation of best model
    best_model = model_eff_ft  # Assuming fine-tuned EfficientNet is best
    best_pred = eff_ft_pred
    best_pred_classes = np.argmax(best_pred, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    print("\nDetailed Classification Report:")
    print(classification_report(true_classes, best_pred_classes,
                              target_names=['Biodegradable', 'Recyclable', 'Non-Recyclable']))

    # Confusion matrix
    cm = confusion_matrix(true_classes, best_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Best Model')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['Biodegradable', 'Recyclable', 'Non-Recyclable'])
    plt.yticks(tick_marks, ['Biodegradable', 'Recyclable', 'Non-Recyclable'])

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

if __name__ == "__main__":
    results = compare_models()
    print("\nFinal Results Summary:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")