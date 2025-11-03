from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    Multiply, Reshape
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def squeeze_excite_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation (SE) Block for channel-wise attention.
    """
    channels = input_tensor.shape[-1]

    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)

    return Multiply()([input_tensor, se])


def build_proposed_model(use_label_smoothing=True, label_smoothing_factor=0.1):
    """
    Build the proposed EfficientNetB0 + SE attention model.
    """
    print("\n[Model Builder] Initializing EfficientNetB0 base...")

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )
    base_model.trainable = False  # freeze for stage 1

    x = base_model.output
    x = squeeze_excite_block(x, ratio=16)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(3, activation='softmax', name='output')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name="Proposed_EfficientNet_SE")

    if use_label_smoothing:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)
    else:
        loss = 'categorical_crossentropy'

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=loss,
        metrics=['accuracy']
    )

    print("✓ Model successfully built.")
    print(f"   Base model frozen layers: {len(base_model.layers)}")
    print(f"   Trainable parameters (stage 1): {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

    return model


def unfreeze_base_model(model, unfreeze_from_layer=50):
    """
    Safely unfreeze the EfficientNetB0 base model for fine-tuning.
    """
    print(f"\n[Fine-Tune] Attempting to unfreeze EfficientNetB0 from layer {unfreeze_from_layer}...")

    # Try to locate EfficientNet base inside the full model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        print("⚠ EfficientNet base not found directly, using global layer list instead.")
        target_layers = model.layers
    else:
        target_layers = base_model.layers
        print(f"✓ Found base model: {base_model.name} with {len(target_layers)} layers.")

    if unfreeze_from_layer >= len(target_layers):
        unfreeze_from_layer = len(target_layers) - 1
        print(f"⚠ Adjusted unfreeze_from_layer to {unfreeze_from_layer} (last layer index).")

    # Unfreeze deeper layers
    for layer in target_layers[unfreeze_from_layer:]:
        layer.trainable = True

    for layer in target_layers[:unfreeze_from_layer]:
        layer.trainable = False

    trainable_count = sum([layer.trainable for layer in model.layers])
    print(f"✅ Successfully unfrozen {len(target_layers[unfreeze_from_layer:])} layers (from {unfreeze_from_layer} onward).")
    print(f"   Total trainable layers in full model: {trainable_count}/{len(model.layers)}")

    # Recompile after changing trainability
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=model.loss,
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_proposed_model()
    print("\n=== Proposed Model Summary ===")
    model.summary(line_length=140)

    print("\n--- Fine-tune test ---")
    model = unfreeze_base_model(model, unfreeze_from_layer=200)
    print(f"Trainable params after unfreezing: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")
