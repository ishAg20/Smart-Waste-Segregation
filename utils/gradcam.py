import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image and model.

    Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes
    which parts of the image the model focuses on for its prediction.

    Args:
        img_array: Preprocessed image array (1, 224, 224, 3)
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Target class index (None = predicted class)

    Returns:
        heatmap: Grad-CAM heatmap (numpy array)
    """
    # Create a model that maps inputs to activations of last conv layer + predictions
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of the predicted class w.r.t. last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the predicted class w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling to get importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of conv layer outputs
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Average over all feature maps
    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalize heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

    return heatmap


def apply_gradcam_overlay(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        img: Original image (H, W, 3) as numpy array or path
        heatmap: Grad-CAM heatmap
        alpha: Transparency of overlay (0-1)
        colormap: OpenCV colormap

    Returns:
        superimposed_img: Image with heatmap overlay
    """
    # Load image if path provided
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure img is in correct format
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Overlay heatmap
    superimposed_img = cv2.addWeighted(
        img,
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )

    return superimposed_img


def get_last_conv_layer_name(model):
    """
    Automatically find the last convolutional layer name in the model.

    Args:
        model: Keras model

    Returns:
        layer_name: Name of last conv layer
    """
    # Try common layer types
    conv_layer_types = [
        'Conv2D', 'SeparableConv2D', 'DepthwiseConv2D',
        'Conv2DTranspose', 'Convolution2D'
    ]

    # Search from end to beginning
    for layer in reversed(model.layers):
        if any(layer_type.lower() in layer.__class__.__name__.lower()
               for layer_type in conv_layer_types):
            return layer.name

    # If not found, try to get from base model (e.g., EfficientNet, MobileNet)
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):  # Check if it's a model (like base_model)
            for sublayer in reversed(layer.layers):
                if any(layer_type.lower() in sublayer.__class__.__name__.lower()
                       for layer_type in conv_layer_types):
                    return sublayer.name

    raise ValueError("Could not find a convolutional layer in the model")


def visualize_gradcam(img_path, model, last_conv_layer_name=None, save_path=None):
    """
    Complete pipeline to visualize Grad-CAM for an image.

    Args:
        img_path: Path to input image
        model: Trained Keras model
        last_conv_layer_name: Name of last conv layer (auto-detected if None)
        save_path: Path to save visualization (optional)

    Returns:
        result_img: Image with Grad-CAM overlay
        prediction: Model prediction
        confidence: Prediction confidence
    """
    # Load and preprocess image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]

    # Auto-detect last conv layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
        print(f"Auto-detected last conv layer: {last_conv_layer_name}")

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
        pred_index=pred_class
    )

    # Apply overlay
    result_img = apply_gradcam_overlay(img_resized, heatmap, alpha=0.4)

    # Save if requested
    if save_path:
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_img_bgr)
        print(f"Grad-CAM visualization saved to {save_path}")

    categories = ["Biodegradable", "Recyclable", "Non-Recyclable"]

    return result_img, categories[pred_class], confidence


if __name__ == "__main__":
    # Example usage
    from tensorflow.keras.models import load_model

    print("Testing Grad-CAM utility...")
    print("\nTo use this utility:")
    print("1. from utils.gradcam import visualize_gradcam")
    print("2. result_img, pred, conf = visualize_gradcam('path/to/image.jpg', model)")
    print("\nGrad-CAM helps visualize what the model 'sees' when making predictions!")
