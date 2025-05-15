import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Data Loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(224, 224),  # EfficientNet prefers 224x224
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_test_datagen.flow_from_directory(
    "data/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Model Architecture
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base_model.trainable = False  # Freeze pretrained layers

model = tf.keras.Sequential([
    base_model,
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(9, activation="softmax")  # 9 classes
])

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint("models/best_model.keras", save_best_only=True),
    TensorBoard(log_dir="logs")
]

# Compile & Train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks
)