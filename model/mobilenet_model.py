from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_model(model_type='mobilenet', fine_tune=False, num_classes=3):
    """
    Build improved model with various architectures and techniques

    Args:
        model_type: 'mobilenet', 'efficientnet', or 'resnet'
        fine_tune: Whether to fine-tune the base model
        num_classes: Number of output classes
    """

    if model_type == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_type == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_type == 'resnet':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("model_type must be 'mobilenet', 'efficientnet', or 'resnet'")

    base_model.trainable = fine_tune

    if fine_tune:
        # Fine-tune only the top layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Use different learning rates for fine-tuning vs feature extraction
    lr = 0.00001 if fine_tune else 0.001
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
