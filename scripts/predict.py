import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("models/best_model.keras")

# Class names (must match training order)
class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Pigmented Benign Keratosis",
    "Seborrheic Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion"
]

def predict_skin_disease(img_path):
    """Predict skin disease from an image."""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))  # Must match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize (same as training)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Display results
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.2%}")
    plt.show()

    return class_names[predicted_class], confidence

# Example usage
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip('"')  # Handle drag-and-drop paths
    class_name, confidence = predict_skin_disease(image_path)
    print(f"\nPredicted Class: {class_name}")
    print(f"Confidence: {confidence:.2%}")