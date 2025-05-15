import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("models/best_model.keras")

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# Detailed report
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(
    test_generator.classes,
    y_pred_classes,
    target_names=list(test_generator.class_indices.keys())
))