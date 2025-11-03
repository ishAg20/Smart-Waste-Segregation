import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from focal_loss import SparseCategoricalFocalLoss
from proposed_model import build_proposed_model, unfreeze_base_model
import tensorflow as tf
from datetime import datetime


def create_augmented_generator(X_train, y_train, batch_size=32, sparse=False):
    """
    Create data generator with augmentation.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1
    )

    train_gen = datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        subset='training'
    )

    val_gen = datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        subset='validation'
    )

    return train_gen, val_gen


def train_proposed_model(
    use_focal_loss=True,
    use_augmentation=True,
    two_stage_training=True,
    epochs_stage1=5,
    epochs_stage2=8
):
    print("=" * 70)
    print("PROPOSED MODEL TRAINING - Smart Waste Segregation")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading preprocessed data...")
    X_train, X_test, y_train, y_test = joblib.load("data_preprocessing/split_data.pkl")
    print(f"✓ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Build model
    print("\n[2/6] Building proposed model...")
    model = build_proposed_model(use_label_smoothing=True, label_smoothing_factor=0.1)
    print("✓ Model built with EfficientNetB0 + SE blocks")

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint('saved_models/proposed_model_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=f'logs/proposed_model_{timestamp}', histogram_freq=1)
    ]

    # ============================================================
    # STAGE 1: Frozen base model
    # ============================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Training with FROZEN base model")
    print("=" * 70)

    if use_augmentation:
        train_gen, val_gen = create_augmented_generator(X_train, y_train)
        history_stage1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_stage1,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history_stage1 = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs_stage1,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

    print("\n✓ Stage 1 complete!")

    # ============================================================
    # STAGE 2: Fine-tuning
    # ============================================================
    if two_stage_training:
        print("\n" + "=" * 70)
        print("STAGE 2: Fine-tuning with UNFROZEN base model")
        print("=" * 70)

        model = unfreeze_base_model(model, unfreeze_from_layer=50)

        if use_focal_loss:
            print("\n[5/6] Switching to Focal Loss for fine-tuning...")
            y_train_sparse = np.argmax(y_train, axis=1)
            y_test_sparse = np.argmax(y_test, axis=1)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss=SparseCategoricalFocalLoss(gamma=2.0),
                metrics=['accuracy']
            )

            if use_augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2,
                    brightness_range=[0.8, 1.2],
                    horizontal_flip=True,
                    fill_mode='nearest',
                    validation_split=0.1
                )

                train_gen = datagen.flow(X_train, y_train_sparse, batch_size=32, subset='training')
                val_gen = datagen.flow(X_train, y_train_sparse, batch_size=32, subset='validation')

                history_stage2 = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=epochs_stage2,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history_stage2 = model.fit(
                    X_train, y_train_sparse,
                    validation_split=0.1,
                    epochs=epochs_stage2,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1
                )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            if use_augmentation:
                train_gen, val_gen = create_augmented_generator(X_train, y_train)
                history_stage2 = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=epochs_stage2,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history_stage2 = model.fit(
                    X_train, y_train,
                    validation_split=0.1,
                    epochs=epochs_stage2,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1
                )

        print("\n✓ Stage 2 complete!")
    else:
        history_stage2 = None

    # ============================================================
    # Final Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("[6/6] Final Evaluation on Test Set")
    print("=" * 70)

    if use_focal_loss:
        y_test_eval = np.argmax(y_test, axis=1)
        test_loss, test_accuracy = model.evaluate(X_test, y_test_eval, verbose=0)
    else:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n✓ Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"✓ Test Loss: {test_loss:.4f}")

    model.save('saved_models/proposed_model_final.h5')
    print("\n✓ Model saved to 'saved_models/proposed_model_final.h5'")

    plot_training_history(history_stage1, history_stage2)


def plot_training_history(history1, history2=None):
    plt.figure(figsize=(14, 5))

    if history2:
        train_acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        train_loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
        stage1_epochs = len(history1.history['accuracy'])
    else:
        train_acc = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        train_loss = history1.history['loss']
        val_loss = history1.history['val_loss']
        stage1_epochs = None

    epochs = range(1, len(train_acc) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    if stage1_epochs:
        plt.axvline(x=stage1_epochs, color='g', linestyle='--', label='Fine-tuning starts')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    if stage1_epochs:
        plt.axvline(x=stage1_epochs, color='g', linestyle='--', label='Fine-tuning starts')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('saved_models/training_history_proposed.png', dpi=150)
    print("\n✓ Training history plot saved to 'saved_models/training_history_proposed.png'")
    plt.close()


if __name__ == "__main__":
    train_proposed_model(
        use_focal_loss=True,
        use_augmentation=True,
        two_stage_training=True
    )
