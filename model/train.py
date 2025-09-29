import joblib
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from mobilenet_model import build_model
from data_preprocessing.preprocess import create_data_augmentation

# Load data
X_train, X_test, y_train, y_test = joblib.load("data_preprocessing/split_data.pkl")

def lr_schedule(epoch):
    """Learning rate scheduler"""
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0001
    else:
        return 0.00001

def train_improved_model(model_type='efficientnet', use_augmentation=True, fine_tune_phase=True):
    """
    Train improved model with multiple phases and techniques
    """

    # Phase 1: Feature extraction
    print("Phase 1: Feature extraction training...")
    model = build_model(model_type=model_type, fine_tune=False, num_classes=3)

    callbacks_phase1 = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(f'saved_models/best_model_{model_type}_phase1.h5',
                       monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    if use_augmentation:
        datagen = create_data_augmentation()
        datagen.fit(X_train)

        history_phase1 = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            validation_data=(X_test, y_test),
            epochs=25,
            callbacks=callbacks_phase1
        )
    else:
        history_phase1 = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=25,
            batch_size=32,
            callbacks=callbacks_phase1
        )

    if not fine_tune_phase:
        return model, history_phase1

    # Phase 2: Fine-tuning
    print("Phase 2: Fine-tuning training...")
    model = build_model(model_type=model_type, fine_tune=True, num_classes=3)
    model.load_weights(f'saved_models/best_model_{model_type}_phase1.h5')

    callbacks_phase2 = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'saved_models/best_model_{model_type}_final.h5',
                       monitor='val_accuracy', save_best_only=True),
        LearningRateScheduler(lr_schedule)
    ]

    if use_augmentation:
        datagen_finetune = create_data_augmentation()
        datagen_finetune.rotation_range = 15  # Reduce augmentation for fine-tuning
        datagen_finetune.zoom_range = 0.1
        datagen_finetune.fit(X_train)

        history_phase2 = model.fit(
            datagen_finetune.flow(X_train, y_train, batch_size=16),
            steps_per_epoch=len(X_train) // 16,
            validation_data=(X_test, y_test),
            epochs=30,
            callbacks=callbacks_phase2
        )
    else:
        history_phase2 = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=16,
            callbacks=callbacks_phase2
        )

    return model, (history_phase1, history_phase2)

if __name__ == "__main__":
    # Train with EfficientNet (generally better performance)
    model, history = train_improved_model(
        model_type='efficientnet',
        use_augmentation=True,
        fine_tune_phase=True
    )
