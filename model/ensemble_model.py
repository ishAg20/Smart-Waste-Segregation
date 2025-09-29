import numpy as np
from tensorflow.keras.models import load_model
from mobilenet_model import build_model

class EnsembleModel:
    """Ensemble of multiple models for improved accuracy"""

    def __init__(self, model_paths=None):
        self.models = []
        self.model_paths = model_paths or []

    def add_model(self, model_path):
        """Add a model to the ensemble"""
        model = load_model(model_path)
        self.models.append(model)
        self.model_paths.append(model_path)

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train multiple models with different configurations"""
        from model.train import train_improved_model

        # Train EfficientNet model
        print("Training EfficientNet model...")
        model1, _ = train_improved_model('efficientnet', use_augmentation=True, fine_tune_phase=True)
        self.models.append(model1)

        # Train ResNet model
        print("Training ResNet model...")
        model2, _ = train_improved_model('resnet', use_augmentation=True, fine_tune_phase=True)
        self.models.append(model2)

        # Train MobileNet model
        print("Training MobileNet model...")
        model3, _ = train_improved_model('mobilenet', use_augmentation=True, fine_tune_phase=True)
        self.models.append(model3)

    def predict(self, X, method='average'):
        """
        Make ensemble predictions

        Args:
            X: Input data
            method: 'average', 'weighted', or 'voting'
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted':
            # Weight models based on validation accuracy (you can customize weights)
            weights = [0.4, 0.35, 0.25]  # EfficientNet, ResNet, MobileNet
            weighted_preds = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_preds += weights[i] * pred
            return weighted_preds
        elif method == 'voting':
            # Hard voting
            class_predictions = np.argmax(predictions, axis=2)
            final_predictions = np.zeros((predictions.shape[1], predictions.shape[2]))
            for i in range(predictions.shape[1]):
                for j in range(predictions.shape[2]):
                    votes = class_predictions[:, i]
                    unique, counts = np.unique(votes, return_counts=True)
                    final_predictions[i, unique[np.argmax(counts)]] = 1
            return final_predictions
        else:
            raise ValueError("method must be 'average', 'weighted', or 'voting'")

    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        predictions = self.predict(X_test, method='weighted')
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy