from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    Multiply, Reshape, Lambda
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def squeeze_excite_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation (SE) Block for channel-wise attention.

    Args:
        input_tensor: Input feature map
        ratio: Reduction ratio for SE block

    Returns:
        Recalibrated feature map
    """
    channels = input_tensor.shape[-1]

    # Squeeze: Global average pooling
    se = GlobalAveragePooling2D()(input_tensor)

    # Excitation: FC -> ReLU -> FC -> Sigmoid
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)

    # Reshape to match input dimensions
    se = Reshape((1, 1, channels))(se)

    # Scale: multiply input with attention weights
    return Multiply()([input_tensor, se])


def build_proposed_model(use_label_smoothing=True, label_smoothing_factor=0.1):
    """
    Build the proposed waste classification model with:
    - EfficientNetB0 as base (better than MobileNetV2)
    - SE (Squeeze-and-Excitation) blocks for attention
    - Deeper classification head
    - Label smoothing for better generalization

    Args:
        use_label_smoothing: Whether to use label smoothing
        label_smoothing_factor: Smoothing factor (0.1 recommended)

    Returns:
        Compiled Keras model
    """
    # Base model: EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Initially freeze base model for transfer learning
    base_model.trainable = False

    # Get base model output
    x = base_model.output

    # Add SE block for attention mechanism
    x = squeeze_excite_block(x, ratio=16)

    # Global pooling
    x = GlobalAveragePooling2D()(x)

    # Dense classification head
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.3)(x)

    # Output layer (3 classes: Biodegradable, Recyclable, Non-Recyclable)
    predictions = Dense(3, activation='softmax', name='output')(x)

    # Build model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile with label smoothing if requested
    if use_label_smoothing:
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing_factor
        )
    else:
        loss = 'categorical_crossentropy'

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=loss,
        metrics=['accuracy']
    )

    return model


def unfreeze_base_model(model, unfreeze_from_layer=50):
    """
    Unfreeze the base model for fine-tuning.

    Args:
        model: The compiled model
        unfreeze_from_layer: Unfreeze layers from this index onwards

    Returns:
        Model with unfrozen base
    """
    # Get the base model (EfficientNetB0)
    base_model = model.layers[0]

    # Unfreeze base
    base_model.trainable = True

    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss=model.loss,
        metrics=['accuracy']
    )

    print(f"Base model unfrozen from layer {unfreeze_from_layer}")
    print(f"Total trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

    return model


if __name__ == "__main__":
    # Test model building
    model = build_proposed_model()
    print("\n=== Proposed Model Summary ===")
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
